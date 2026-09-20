"""Labelled-chip dataset contract for adapting the crop-classification model: the pinned multi-temporal crop
sample, role assignment by spatial block, BYOD loaders and sample export.

The default dataset is **real**: 60 labelled 224 × 224 chips of the HLS multi-temporal crop classification dataset
(NASA IMPACT / IBM, CC BY 4.0) — three HLS dates of six bands over the contiguous United States in 2022, with a
13-class label derived from the USDA Cropland Data Layer — drawn on 2026-09-20 from the 771 chips of the
`validation_chips.tgz` archive (each an image with its mask). Every one of the 60 chips was part of the upstream
*validation* split, i.e. the split the published checkpoint was selected on (`best_mIoU_epoch_80`), so the frozen
model has seen these chips as validation data but was never trained on them. Roles are assigned per 4 × 4-chip
block of the chip grid (a seeded hash of the block; 36 train / 12 validation / 12 test, each stratified by dominant
class so all 13 classes occur in every role) — chips of one block never straddle roles, but neighbouring blocks
may, so the split is by block, not by region; real data must be split by region. The dataset is distributed as one
1.18 GB gzipped tarball on the Hugging Face Hub; the tarball is pinned by byte size and SHA-256, each pinned member
is pinned again by size and SHA-256 and extracted **without** `extractall` into the cache, and everything else in
the archive (including its macOS `._` resource-fork twins) is left alone. The repository redistributes none of
the chips.

A record is ``{id, image, label}``: a (3, 6, 224, 224) digital-number array (or an 18-band GeoTIFF path) and a
(224, 224) mask with classes 0..12 and -1 = no data (or a GeoTIFF path with 0 = no data, 1..13 = class).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import (
    BANDS,
    IGNORE_INDEX,
    IMAGE_SIZE,
    MIN_RECORDS,
    MODEL_ID,
    NUM_FRAMES,
    check_record,
    chip_digest,
    read_chip,
    read_mask,
    validate_dataset,
)

CORPUS_NAME = "HLS multi-temporal crop classification chips (NASA IMPACT / IBM)"
CORPUS_RELEASE = (
    "Hugging Face dataset ibm-nasa-geospatial/multi-temporal-crop-classification, 60 chips selected 2026-09-20 from "
    "the validation archive"
)
CORPUS_LICENSE = "CC BY 4.0 (NASA IMPACT / IBM)"
DATASET_ID = "ibm-nasa-geospatial/multi-temporal-crop-classification"
DATASET_REVISION = "f285bb27c8f623a0fb6a44a6fd953c3ad34007d6"
CORPUS_BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/"
TAR_NAME = "validation_chips.tgz"
TAR_BYTES = 1_179_542_384
TAR_SHA256 = "d6e616cc008858a1935a8937e0cdf852d574754273b0655fb696bd29aebd2fd3"
CORPUS_BYTES = 111_617_880  # the 120 pinned members, uncompressed
DEFAULT_CACHE_DIR = Path("weights") / "multi-temporal-crop"
ROLES = ("train", "validation", "test")
BLOCK_SIZE = 4  # chips per side of the grid blocks that roles are assigned by
# (chip key, role by block, image member, image bytes, image sha256, mask member, mask bytes, mask sha256) —
# member paths are relative to the tarball root
SAMPLE_RECORDS: tuple[tuple[str, str, str, int, str, str, int, str], ...] = (
    (
        "chip_065_352",
        "train",
        "validation_chips/chip_065_352_merged.tif",
        1808174,
        "e63542cdc0bcf0fc2c5ae4befc7c0613cf823ca131b7ec23754847b1636a6b4d",
        "validation_chips/chip_065_352.mask.tif",
        52124,
        "fb02241466945389394aeb5f33883e06c155d0bab77f6548dcb0e1f08dce0eba",
    ),
    (
        "chip_084_363",
        "train",
        "validation_chips/chip_084_363_merged.tif",
        1808174,
        "f857f6a8e5bcb358e773539fd7bf800a49c622cfb94cacebf970be2aa95e63ae",
        "validation_chips/chip_084_363.mask.tif",
        52124,
        "e39142ceaa5a05d8b084fd15576e3808c499f4f0bcd8cfcba83ee0b7164133a9",
    ),
    (
        "chip_094_337",
        "train",
        "validation_chips/chip_094_337_merged.tif",
        1808174,
        "ce9d684437c40b9f9fa964f99d598216aff1e3dc8bb26e1ada60f22835d30eec",
        "validation_chips/chip_094_337.mask.tif",
        52124,
        "7ff0e18a1625b91365e6f7f60967334da1b95bbd89c8304e9ddc5cf22ece5a3a",
    ),
    (
        "chip_096_341",
        "train",
        "validation_chips/chip_096_341_merged.tif",
        1808174,
        "3fea1ffe7d51fd1484a5b8004f54c284f2bd0b2fbadce7c8103496e406e0e3f6",
        "validation_chips/chip_096_341.mask.tif",
        52124,
        "34693ae6da7ae187ea747e4f0fc2a1be18f2761ce8207b8cb970296eab2fffd9",
    ),
    (
        "chip_106_441",
        "train",
        "validation_chips/chip_106_441_merged.tif",
        1808174,
        "059726da4c85e6e828d4cba6ac1c846411023200f376e002af5b680db8b0e01e",
        "validation_chips/chip_106_441.mask.tif",
        52124,
        "09a996cca65c15ecf96242505282609f94bd1501e2a401373cb77a06c5da8398",
    ),
    (
        "chip_120_555",
        "train",
        "validation_chips/chip_120_555_merged.tif",
        1808174,
        "b70f53aad07e1a9b7864059e7220d3bcdc30d74ea9f71a0a9d79af49d8a79ba6",
        "validation_chips/chip_120_555.mask.tif",
        52124,
        "f5041034c3bb9b720b96ec73db30ad3a0a06f0e3ba3207521fa4d1048fb82d70",
    ),
    (
        "chip_128_312",
        "train",
        "validation_chips/chip_128_312_merged.tif",
        1808174,
        "68b22054f2509b182466b0b15f338cac442c8c34d17f5a1eee6accf21ecd7ee0",
        "validation_chips/chip_128_312.mask.tif",
        52124,
        "4723fe0c1913b0357a101fdc1655f71076d2df4c6b9d3fcad9a443a464dad7af",
    ),
    (
        "chip_128_482",
        "train",
        "validation_chips/chip_128_482_merged.tif",
        1808174,
        "9c3f6ad7b8efdca2dd89c685d038eb7c6daaa8bba1f60aec29a50e99315217b7",
        "validation_chips/chip_128_482.mask.tif",
        52124,
        "fae18d2a7489af82bb1e4de5412592cf202b5a3592a1ffeefa24aa51106df225",
    ),
    (
        "chip_135_484",
        "train",
        "validation_chips/chip_135_484_merged.tif",
        1808174,
        "9f06e3c8409006319f6bec2671d863355ea8e7179ed839772d8872b557aa0fe4",
        "validation_chips/chip_135_484.mask.tif",
        52124,
        "d8f1ebd287fe29504b687ff1de49c8fcf38dd950887a366942326d9e8e698e2e",
    ),
    (
        "chip_144_487",
        "train",
        "validation_chips/chip_144_487_merged.tif",
        1808174,
        "ae230d8ecb1e198876d481ef3588377040528458e98ddfb41ecd6d74dc1e4a7b",
        "validation_chips/chip_144_487.mask.tif",
        52124,
        "8158fa57455349a1babae69f939399cb957bddaf14ff19036aa3c12668de94c2",
    ),
    (
        "chip_165_447",
        "train",
        "validation_chips/chip_165_447_merged.tif",
        1808174,
        "5423cebb6024d283773f3b6c664529b5b30b7a3fb64af1581d190ca5a97846ae",
        "validation_chips/chip_165_447.mask.tif",
        52124,
        "7ad45688714d11d0e774dfaaf4a196e56eecefe92075c2e8f4670284fcdc1026",
    ),
    (
        "chip_168_274",
        "train",
        "validation_chips/chip_168_274_merged.tif",
        1808174,
        "9deaaf3dd88ca854b6419257d7d728352b64a1361a20d8139586812bf3ad2835",
        "validation_chips/chip_168_274.mask.tif",
        52124,
        "2fb56932d9b3111d23bee6d7327b1c59708e9731e445b2657d3261ee4f880633",
    ),
    (
        "chip_173_350",
        "train",
        "validation_chips/chip_173_350_merged.tif",
        1808174,
        "576eb70cd3987677d96bc2d859022868a7bdbd32879a7589d0a947f719f16085",
        "validation_chips/chip_173_350.mask.tif",
        52124,
        "9b5f98df54cbecac6dc0a0b6df7ae7f2f6224b87c49e7d89bb9659179ee53076",
    ),
    (
        "chip_178_022",
        "train",
        "validation_chips/chip_178_022_merged.tif",
        1808174,
        "c5bd551f640d1abf8f6285ea190f3657132c74189a9f5c37a83d7c813d67df6b",
        "validation_chips/chip_178_022.mask.tif",
        52124,
        "1d2ae12bc1bebe9e527607dd1df74dfd710a011fc7069655a6c359bfd467f986",
    ),
    (
        "chip_184_356",
        "train",
        "validation_chips/chip_184_356_merged.tif",
        1808174,
        "9cc45364c0f7613f0ede41e1bee6921ad6cb9e9bc1feb9eece99e785d8aa9bd3",
        "validation_chips/chip_184_356.mask.tif",
        52124,
        "bf080397034272153049e2eb57dc2b3b9c86948fcd370321e0409a7de93c0109",
    ),
    (
        "chip_199_425",
        "train",
        "validation_chips/chip_199_425_merged.tif",
        1808174,
        "33fff6b9e9114439285e59317ff6e932436fc7864acac6ef634eb54361b58fb7",
        "validation_chips/chip_199_425.mask.tif",
        52124,
        "e7c2fc669f8f64e1cfefbd6d8b7853b00aae9bd7c150d932134a41419fb4d25d",
    ),
    (
        "chip_219_323",
        "train",
        "validation_chips/chip_219_323_merged.tif",
        1808174,
        "17262af4c3aba66de068dc5da1466c3a93e7a7ef255e1d1c5fae15f928442331",
        "validation_chips/chip_219_323.mask.tif",
        52124,
        "b2a2810301afd0565a6de77dcc156dd4ccb5d27aea2292959f749c8544ca866f",
    ),
    (
        "chip_220_286",
        "train",
        "validation_chips/chip_220_286_merged.tif",
        1808174,
        "54ac423ddcc4bd6a8abe7cc8bc78e90249651f41e41d7f5399b52998e523c826",
        "validation_chips/chip_220_286.mask.tif",
        52124,
        "7b6af94771bccf9d1ffd8ccdcfcbde542b1d9bf010cdf25062bf8650ee39ca77",
    ),
    (
        "chip_221_442",
        "train",
        "validation_chips/chip_221_442_merged.tif",
        1808174,
        "c9c3772f2f4aabf0c52a3a83b43a9129db92a56d785fbaa6cea54dd317ab2c64",
        "validation_chips/chip_221_442.mask.tif",
        52124,
        "22c17b696b2d83bff852ae4221dfcf93fe4e7b30ac57a39c5fa14407b47414eb",
    ),
    (
        "chip_227_432",
        "train",
        "validation_chips/chip_227_432_merged.tif",
        1808174,
        "e836131999d42acd9e4517a0f36f43ce6c2d0bdc7bbc6bdfc6a5fd38f741339b",
        "validation_chips/chip_227_432.mask.tif",
        52124,
        "ee46977d225def49158ba2fb77e2864025a0766afdeafb7c566633faf25f5b5e",
    ),
    (
        "chip_229_293",
        "train",
        "validation_chips/chip_229_293_merged.tif",
        1808174,
        "ed6154007d0b01694d866525c4930b7bc82da060504c94dd7f0ef9a234283db3",
        "validation_chips/chip_229_293.mask.tif",
        52124,
        "dfc3880a43155fc9dface5e3c5db82b9836829b87e48b6c1b0044ed8443d7195",
    ),
    (
        "chip_231_274",
        "train",
        "validation_chips/chip_231_274_merged.tif",
        1808174,
        "a44f14f44cbb3f5dcc6aac0fd683df56f1b19be03b6e4ca064decb501a572b73",
        "validation_chips/chip_231_274.mask.tif",
        52124,
        "f33da9adee744d682295f869de6d6d56e4924100062cdf4bd984139d077e228b",
    ),
    (
        "chip_231_595",
        "train",
        "validation_chips/chip_231_595_merged.tif",
        1808174,
        "5527a4e81ae8e974308ee9e8e1d3c8a252cc5a3cad329d8434df9f5c217217d2",
        "validation_chips/chip_231_595.mask.tif",
        52124,
        "2c475d72b4de9b79de039a82efedd0635b1d94c70ca880e6170c1400e9a98dbf",
    ),
    (
        "chip_232_279",
        "train",
        "validation_chips/chip_232_279_merged.tif",
        1808174,
        "fd7133455516f1b16af93bfff02eb64fd08e587f2b871db529241f73e977d094",
        "validation_chips/chip_232_279.mask.tif",
        52124,
        "eb91554cc9f02942cf54a5bf3d8c532a344e103dbc34939a5b4bb614c48e785b",
    ),
    (
        "chip_236_335",
        "train",
        "validation_chips/chip_236_335_merged.tif",
        1808174,
        "c3e491c8bdd8330d99a25c3f53f51f081ccff2fc50a942432e33ebb8e2d2148b",
        "validation_chips/chip_236_335.mask.tif",
        52124,
        "62c9a6d795521caf6991a35660a03c1e438d9407bbf313b2b351601029964315",
    ),
    (
        "chip_243_450",
        "train",
        "validation_chips/chip_243_450_merged.tif",
        1808174,
        "aab5c153a372acfabadb5a2fc11d19b0a9d2c1209507bcf0b62e29fd39efcba9",
        "validation_chips/chip_243_450.mask.tif",
        52124,
        "1561dc15054881629378e8be82cb169d16c63f2dc54e8e475dc5e7ca2b9961ff",
    ),
    (
        "chip_247_333",
        "train",
        "validation_chips/chip_247_333_merged.tif",
        1808174,
        "3e5de54fec956202585b3230d6bacb290625976e142b04c6f781aa5cafcaefde",
        "validation_chips/chip_247_333.mask.tif",
        52124,
        "fc589c9f90770c2c37e6f7d8538f34c154c0a6b059e8a8d6ce1548e2b6a2ccbe",
    ),
    (
        "chip_253_285",
        "train",
        "validation_chips/chip_253_285_merged.tif",
        1808174,
        "abd09ec6abe9c37f69a4eebffee548128c143c943140c77258d98fead6e4752a",
        "validation_chips/chip_253_285.mask.tif",
        52124,
        "ec4089fcc6be711c0d6145d368e9db2c118ba7d948393eeddd29aa5646a251be",
    ),
    (
        "chip_254_285",
        "train",
        "validation_chips/chip_254_285_merged.tif",
        1808174,
        "007a63570ce62518f777092c2d1b02761f005d2232852ec6ca3f711190eac148",
        "validation_chips/chip_254_285.mask.tif",
        52124,
        "21f92de44b1d44a13a6b9909fd1c1e423d7053ecd76dcf035c884395d7f74453",
    ),
    (
        "chip_273_471",
        "train",
        "validation_chips/chip_273_471_merged.tif",
        1808174,
        "65c83afb05e71f60e3d400266aac15a79686771099e184c11592d1fb2dc009ce",
        "validation_chips/chip_273_471.mask.tif",
        52124,
        "7d6e3e41b0da5f59ae9f8e54d4ddab0bc7e79b3fda2216dc11d769faebd9bc42",
    ),
    (
        "chip_281_251",
        "train",
        "validation_chips/chip_281_251_merged.tif",
        1808174,
        "87901340e008a063702d0981d2016d40c8da155c8f01f592e037f8bf20e18882",
        "validation_chips/chip_281_251.mask.tif",
        52124,
        "60a0fa571645842958eed436aaa6abfa341a4cfe98807922b8afd9f7b1b71668",
    ),
    (
        "chip_282_404",
        "train",
        "validation_chips/chip_282_404_merged.tif",
        1808174,
        "706ea131325106696cebb5ba2105852a30c1cea47903c47a313e1aac1fddeddb",
        "validation_chips/chip_282_404.mask.tif",
        52124,
        "fc6cfdcabd4f2b501bafbd5507c5185d74c7698dd8a355d2da3246aec3acc27b",
    ),
    (
        "chip_283_264",
        "train",
        "validation_chips/chip_283_264_merged.tif",
        1808174,
        "d5f3f8a399230ad6f16a53706e9ba0d786b2f700dcacc80eaeba4a6ee6110daa",
        "validation_chips/chip_283_264.mask.tif",
        52124,
        "51d0236d693a38d8b4339dd490fa4e0dde78b853912386ce3fce988691c67768",
    ),
    (
        "chip_330_414",
        "train",
        "validation_chips/chip_330_414_merged.tif",
        1808174,
        "915dbf2ac5c8dcc6942b06d29627467879d0394c89d8bbe5696e3c22dfb2e9aa",
        "validation_chips/chip_330_414.mask.tif",
        52124,
        "7e27ab440089e181061146b994f36fe1ce401235df0514c5914115e4e874d0ea",
    ),
    (
        "chip_330_510",
        "train",
        "validation_chips/chip_330_510_merged.tif",
        1808174,
        "7917ecde092fde069ca8625dd7f3b84b3af83ca82db9d04641bb208971e38004",
        "validation_chips/chip_330_510.mask.tif",
        52124,
        "80153c1ab8d6b114480aed92ea1d6adbdc5a284dccfd34a2f3001e29dc32044d",
    ),
    (
        "chip_330_513",
        "train",
        "validation_chips/chip_330_513_merged.tif",
        1808174,
        "f5fb24d8cdadf8df7c3cd9ac917d43eed555d8d9c9b935fd5776470a8d172688",
        "validation_chips/chip_330_513.mask.tif",
        52124,
        "88d5869d50a96ccf983d575591f9b089082df6363676c675557861af49051d9a",
    ),
    (
        "chip_052_043",
        "validation",
        "validation_chips/chip_052_043_merged.tif",
        1808174,
        "b338d72d7fc657aafe0b0c6856db1b22fa76106b1f05c3bc74df259b22c1f40d",
        "validation_chips/chip_052_043.mask.tif",
        52124,
        "f9882ee1f249873fc76f3d2ecc52c4ddb50b5de299b85ceadec53653b248fb7a",
    ),
    (
        "chip_114_438",
        "validation",
        "validation_chips/chip_114_438_merged.tif",
        1808174,
        "6b1efb1c1d5406985babb9296d48206e05824571286354fbfb8c4fd6a5634b79",
        "validation_chips/chip_114_438.mask.tif",
        52124,
        "d9454c30e6896a0918efa2bd91b6d214cb897426413dec6394634a12690c1854",
    ),
    (
        "chip_117_501",
        "validation",
        "validation_chips/chip_117_501_merged.tif",
        1808174,
        "84c284d780feaaaa411f9e8e7fe860ddcfd5f5b90112a0c31a7c1cc452e18a0b",
        "validation_chips/chip_117_501.mask.tif",
        52124,
        "ce315df5853cf94c0fbf4de86148a868a523167fd6a46be7f2e350e2b12e1cdf",
    ),
    (
        "chip_126_303",
        "validation",
        "validation_chips/chip_126_303_merged.tif",
        1808174,
        "1e8984aa6634c2b18a9eefff02fd46b6c0b693b5ecb89daf49f13e4c50b72d38",
        "validation_chips/chip_126_303.mask.tif",
        52124,
        "9e1ecfec5cd0c77697c7a071d810e5f8f322eddd834ff2d4d608a2c5cb4eadb1",
    ),
    (
        "chip_126_506",
        "validation",
        "validation_chips/chip_126_506_merged.tif",
        1808174,
        "10dda5c3901e5ad4c6efad26f56578131f5ab2fdf010f9ea4f1d215d2e7f614a",
        "validation_chips/chip_126_506.mask.tif",
        52124,
        "1f03e9f71e4b60ba353026df22d5000a9ce9e53de4846f5c56dd98fdf37535ac",
    ),
    (
        "chip_156_476",
        "validation",
        "validation_chips/chip_156_476_merged.tif",
        1808174,
        "5e76c735dbcb10676e1b025ba69fef07fc6be97df8efb99438693db160e76b8a",
        "validation_chips/chip_156_476.mask.tif",
        52124,
        "05c45dacc89bfd17d50759a0d92e0c5cf34da0f4239ed55773c0db05b29d28e9",
    ),
    (
        "chip_213_300",
        "validation",
        "validation_chips/chip_213_300_merged.tif",
        1808174,
        "13399cb90a61f0bba5f7aecae0b3195bfc75e911b5f3b812312c32d3a3a423b5",
        "validation_chips/chip_213_300.mask.tif",
        52124,
        "3dadc16d85882be33af1b74461b18ef1a0c6428b02e7de6bbf5a8d5abe25bfcc",
    ),
    (
        "chip_221_307",
        "validation",
        "validation_chips/chip_221_307_merged.tif",
        1808174,
        "63b268cbc42c48398d7b178ccd72b1cca2e73e36edd7d7475ca0d716c655a1a0",
        "validation_chips/chip_221_307.mask.tif",
        52124,
        "07fc0528c184cb75cd47218d5acdbc9a8948c39f24f532674729cd25793e6692",
    ),
    (
        "chip_225_326",
        "validation",
        "validation_chips/chip_225_326_merged.tif",
        1808174,
        "e442c927ffc0978ca02c382b747a5ab494e69eaace0389808a0385fdebbfdb36",
        "validation_chips/chip_225_326.mask.tif",
        52124,
        "0370a05b4d24398c2f51b829376d7991600ddc3b2d2f30451b3282d232d4e5fd",
    ),
    (
        "chip_227_290",
        "validation",
        "validation_chips/chip_227_290_merged.tif",
        1808174,
        "6eec42de00aab30a1047197416b4c78087ec9fc228cb62ee38d3116d61bcfc23",
        "validation_chips/chip_227_290.mask.tif",
        52124,
        "33283b5da5d113030e98ceeed016a5b9007ed3d216515b643dacc2014f2f2633",
    ),
    (
        "chip_261_443",
        "validation",
        "validation_chips/chip_261_443_merged.tif",
        1808174,
        "f2042785219fb8580f41df877ed37a79ab5ab7aadd8695fccc9f0c70afbefc88",
        "validation_chips/chip_261_443.mask.tif",
        52124,
        "0eaa82b38f6c882e2d72d2b275f425a74b98195c5bbc2a6dc99cfcfcadd95331",
    ),
    (
        "chip_334_470",
        "validation",
        "validation_chips/chip_334_470_merged.tif",
        1808174,
        "5cf931f6675cb496269f1c47d8fa6c99fbd30e27f43aa4a458e7f0155a65b698",
        "validation_chips/chip_334_470.mask.tif",
        52124,
        "85f585f88997025c7ac9a9cb7b621b6de819facab41641d6075f9d0e19d59141",
    ),
    (
        "chip_061_092",
        "test",
        "validation_chips/chip_061_092_merged.tif",
        1808174,
        "02aac755376678f7eaee2fa78b463867342899b26c4b41146e3b962232c09175",
        "validation_chips/chip_061_092.mask.tif",
        52124,
        "b46159e87e0f92ab05bc79dae49dce6898512461aa301b791cecc5701c9fe301",
    ),
    (
        "chip_104_104",
        "test",
        "validation_chips/chip_104_104_merged.tif",
        1808174,
        "c1531d1489b1e24bbf0debccd6992c417e0a2d0dbeec6fa10b7fefa93c029581",
        "validation_chips/chip_104_104.mask.tif",
        52124,
        "764c0c49f2297a78b7ead1b667b015aab55d1ec919f67bfb70e9af763d5e1ce3",
    ),
    (
        "chip_131_433",
        "test",
        "validation_chips/chip_131_433_merged.tif",
        1808174,
        "80c8e613b52fef3e330824d6b64e9b567e8cdead79a93018e82b04552b0a2617",
        "validation_chips/chip_131_433.mask.tif",
        52124,
        "90cc7b77573a30fd5f7ef634207816ca8e314992bfc22a9588a144cdc43ee5a8",
    ),
    (
        "chip_138_430",
        "test",
        "validation_chips/chip_138_430_merged.tif",
        1808174,
        "5116cc251db483fdc539c1c8a9dced4139279f4e096f7a9046bda5aa7edee1a2",
        "validation_chips/chip_138_430.mask.tif",
        52124,
        "48717e3d5b1c47d0f6d3e587f79588b11e0ea63f8d94c90bb588c05ad20d3d59",
    ),
    (
        "chip_164_477",
        "test",
        "validation_chips/chip_164_477_merged.tif",
        1808174,
        "cc4613243afab7af6d9a1bc8af5924f82061a3cf51604ec78208d119fe55c730",
        "validation_chips/chip_164_477.mask.tif",
        52124,
        "477555a10370922f92a3ad2a1e62051089f9235795236bc379bfb394e7585ec5",
    ),
    (
        "chip_186_422",
        "test",
        "validation_chips/chip_186_422_merged.tif",
        1808174,
        "2d7417d78c7411b3f06794847604fda79d95fc0c5e1600a628419ddeadf93a3b",
        "validation_chips/chip_186_422.mask.tif",
        52124,
        "e7f31a9071d356da27a0e80a2c8d57c278978fa8f51deac552e2badcf349f345",
    ),
    (
        "chip_200_318",
        "test",
        "validation_chips/chip_200_318_merged.tif",
        1808174,
        "4836763ec196b76bfe7938159b88c83fb0966570f9c1621826d1c51d165ac71a",
        "validation_chips/chip_200_318.mask.tif",
        52124,
        "4e90c257ba3e16945adabe8ae23305ecc96218a754ef28827de8b87ccaa3250c",
    ),
    (
        "chip_252_597",
        "test",
        "validation_chips/chip_252_597_merged.tif",
        1808174,
        "d729a03a9f59bbeb94682086736c91345f30c5c498a71e198f8f5bf73af00c84",
        "validation_chips/chip_252_597.mask.tif",
        52124,
        "fe80a448310064b60fc23882c12f4615f58d717fc0ae83c2166387616e31b54b",
    ),
    (
        "chip_281_252",
        "test",
        "validation_chips/chip_281_252_merged.tif",
        1808174,
        "01ceb90805f66b33373ed0e862b8d087444f663b717ff9682be09d5635443bfe",
        "validation_chips/chip_281_252.mask.tif",
        52124,
        "56016f8cfea95058bdb29f0f0a1900d99f93588534c1fdcb880ba2a862a4ecb4",
    ),
    (
        "chip_292_455",
        "test",
        "validation_chips/chip_292_455_merged.tif",
        1808174,
        "4c30f264f376fa5b1886b20c0c4607b45cee9cd9dda6681293cfe56dfda8a071",
        "validation_chips/chip_292_455.mask.tif",
        52124,
        "5b7b10ba70cbdff203703d3f17f002b0a9afb05b7cd7c547a6c4a4fbd4c5d926",
    ),
    (
        "chip_331_527",
        "test",
        "validation_chips/chip_331_527_merged.tif",
        1808174,
        "4dfdbc11b64f69d94fb03e3f6d0366c99e4dd43baf681fa35db6228d7948fcf5",
        "validation_chips/chip_331_527.mask.tif",
        52124,
        "32a93ad0269eb0a278c723856e049065e00769bfa03882a5946da48907fc5dce",
    ),
    (
        "chip_420_326",
        "test",
        "validation_chips/chip_420_326_merged.tif",
        1808174,
        "50e547176a760fa218155e336277d797c5b7b0a746768f89a38ba710df5672e2",
        "validation_chips/chip_420_326.mask.tif",
        52124,
        "7a1114ccf449bcd5ddaee33750bcf6b2b3f873290fb954fc14b3bff871adf249",
    ),
)

SAMPLE_LABEL_SOURCE = f"{CORPUS_NAME}; {CORPUS_RELEASE}; {CORPUS_LICENSE}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chip_block(key: str, *, block_size: int = BLOCK_SIZE) -> str:
    """The grid block a chip key (`chip_<row>_<col>`) falls in, e.g. `r02c17` for block row 2, block column 17."""
    _prefix, row, col = key.split("_")
    return f"r{int(row) // block_size:02d}c{int(col) // block_size:02d}"


def _pinned_members() -> dict[str, tuple[int, str]]:
    out = {}
    for _name, _role, image, image_bytes, image_sha, label, label_bytes, label_sha in SAMPLE_RECORDS:
        out[image] = (image_bytes, image_sha)
        out[label] = (label_bytes, label_sha)
    return out


def _hub_download_tarball(destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(DATASET_ID, TAR_NAME, repo_type="dataset", revision=DATASET_REVISION, local_dir=str(destination.parent))


def fetch_tarball(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> Path:
    """The pinned dataset tarball in the cache, fetched from the Hub at the immutable revision when absent, and
    refused on a size or SHA-256 mismatch (the 1.18 GB file is hashed once per call)."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / TAR_NAME
    if not local.is_file() or local.stat().st_size != TAR_BYTES:
        if fetcher is not None:
            local.write_bytes(fetcher(CORPUS_BASE_URL + TAR_NAME))
        else:
            _hub_download_tarball(local)
    size = local.stat().st_size
    digest = _sha256_file(local)
    if size != TAR_BYTES or digest != TAR_SHA256:
        raise ValueError(f"{TAR_NAME}: {size} bytes with sha256 {digest[:16]}…, pinned {TAR_BYTES} / {TAR_SHA256[:16]}…")
    return local


def extract_pinned_members(tar_path: str | Path, *, cache_dir: str | Path | None = None) -> dict[str, bytes]:
    """Stream through the tarball once and copy out exactly the pinned members (no `extractall`, no paths from
    the archive: each is written under its base name in `cache_dir/chips/`), refusing a size or digest mismatch."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    chips = cache / "chips"
    chips.mkdir(parents=True, exist_ok=True)
    wanted = _pinned_members()
    out: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            if member.name not in wanted or not member.isfile():
                continue
            size, sha = wanted[member.name]
            handle = archive.extractfile(member)
            data = handle.read() if handle is not None else b""
            if len(data) != size or _sha256_bytes(data) != sha:
                raise ValueError(
                    f"{member.name}: {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, pinned {size} / {sha[:16]}…"
                )
            (chips / Path(member.name).name).write_bytes(data)
            out[member.name] = data
    missing = sorted(set(wanted) - set(out))
    if missing:
        raise ValueError(f"tarball does not contain {len(missing)} pinned members, e.g. {missing[:3]}")
    return out


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, dict[str, bytes]]:
    """Every pinned chip's image and mask bytes, keyed by chip key: from the extracted cache when every file is
    present with its pinned digest, otherwise from the (verified) tarball."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    chips = cache / "chips"
    wanted = _pinned_members()
    cached: dict[str, bytes] = {}
    for member, (size, sha) in wanted.items():
        local = chips / Path(member).name
        if local.is_file() and local.stat().st_size == size:
            data = local.read_bytes()
            if _sha256_bytes(data) == sha:
                cached[member] = data
    if len(cached) != len(wanted):
        cached = extract_pinned_members(fetch_tarball(cache_dir=cache, fetcher=fetcher), cache_dir=cache)
    out = {}
    for name, _role, image, *_rest in SAMPLE_RECORDS:
        label = _rest[2]
        out[name] = {"image": cached[image], "label": cached[label]}
    return out


def read_corpus(files: Mapping[str, Mapping[str, bytes]]) -> dict[str, list[dict[str, Any]]]:
    """Decode the verified bytes into `{id, image, label}` records grouped by role (train / validation / test)."""
    import tempfile

    splits: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    for name, role, image_member, *_rest in SAMPLE_RECORDS:
        if name not in files:
            raise ValueError(f"corpus is missing {name}")
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.tif"
            label_path = Path(tmp) / "label.tif"
            image_path.write_bytes(files[name]["image"])
            label_path.write_bytes(files[name]["label"])
            image = read_chip(image_path)
            label = read_mask(label_path)
        raw = {
            "id": f"{role}-{len(splits[role]):03d}",
            "source_id": name,
            "region": chip_block(name),
            "split": role,
            "image": image,
            "label": label,
            "source": f"{CORPUS_BASE_URL}{TAR_NAME}#{image_member}",
        }
        splits[role].append(check_record(raw))  # range checked, label checked
    return splits


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus (roles by spatial block)."""
    return read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no chip (by pixel digest) and no block (by `region`) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    blocks: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = chip_digest(record)
            if key in seen and seen[key] != name:
                raise ValueError(f"chip {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
            region = record.get("region")
            if region:
                if region in blocks and blocks[region] != name:
                    raise ValueError(f"block {region!r} has chips in both {blocks[region]} and {name}")
                blocks[region] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train / validation / test after de-duplicating chips. Chips of one
    field or one scene are near-duplicates; group them yourself (one region per split) when that matters."""
    import random

    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = chip_digest(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = round(len(unique) * val_fraction)
    splits = {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(f"split leaves {len(splits['train'])} training chips; at least {MIN_RECORDS} are required")
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, label}` records from a directory or a zip holding `pairs.csv` (columns `id`, `image`,
    `label`) beside 18-band 224 × 224 GeoTIFF chips (three dates × six bands, date-major) and single-band label
    rasters (0 = no data, 1..13 = class); files are decoded from bytes, never extracted to disk."""
    import tempfile

    source = Path(path)
    if source.is_dir():
        table = (source / "pairs.csv").read_text(encoding="utf-8")
        loader = lambda name: (source / name).read_bytes()  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "pairs.csv" not in members:
            raise ValueError("BYOD zip must contain pairs.csv")
        table = archive.read(members["pairs.csv"]).decode("utf-8")
        loader = lambda name: archive.read(members[name])  # noqa: E731
    else:
        raise ValueError("BYOD datasets must be a directory or a .zip holding pairs.csv and the GeoTIFF files")
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "image", "label"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"pairs.csv is missing columns {sorted(missing)}")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            image_path = Path(tmp) / "image.tif"
            image_path.write_bytes(loader(row["image"]))
            record: dict[str, Any] = {"id": row["id"], "image": read_chip(image_path)}
            if row.get("label"):
                label_path = Path(tmp) / "label.tif"
                label_path.write_bytes(loader(row["label"]))
                record["label"] = read_mask(label_path)
            out.append(record)
    return out


def write_sample_pair(record: Mapping[str, Any], image_path: str | Path, label_path: str | Path) -> dict[str, str]:
    """Write one record as an 18-band int16 TIFF (date-major, digital numbers) and a single-band uint8 TIFF in the
    dataset's label convention (0 = no data, 1..13 = class) — the BYOD shape, without georeferencing — and
    return both paths."""
    import numpy as np
    import tifffile

    image_out, label_out = Path(image_path), Path(label_path)
    image_out.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(record["image"], dtype=np.float32).reshape(NUM_FRAMES * len(BANDS), IMAGE_SIZE, IMAGE_SIZE)
    tifffile.imwrite(image_out, np.rint(image).astype(np.int16), photometric="minisblack", planarconfig="separate")
    label = np.asarray(record["label"], dtype=np.int64)
    raw = np.where(label == IGNORE_INDEX, 0, label + 1).astype(np.uint8)
    tifffile.imwrite(label_out, raw, photometric="minisblack")
    return {"image": str(image_out), "label": str(label_out)}


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the pairs table of a split (id, image, label, provenance) in the shape BYOD expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image", "label", "region", "source"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "image": f"{record.get('source_id', record['id'])}_merged.tif",
                    "label": f"{record.get('source_id', record['id'])}.mask.tif",
                    "region": record.get("region", ""),
                    "source": record.get("source", ""),
                }
            )
    return out


def dataset_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Validate every split and summarise the dataset (counts, class balance, digests) for provenance exports."""
    summary: dict[str, Any] = {"model_id": MODEL_ID, "image_size": IMAGE_SIZE, "dates": NUM_FRAMES, "splits": {}}
    for name, records in splits.items():
        report = validate_dataset(records, min_records=1)
        summary["splits"][name] = {
            "n_records": report["n_records"],
            "class_pixel_fraction": report["class_pixel_fraction"],
            "classes_present": report["classes_present"],
            "ignored_pixels": report["ignored_pixels"],
            "regions": sorted({str(r.get("region", "")) for r in records if r.get("region")}),
            "digest": report["digest"],
        }
    summary["disjoint"] = check_split_disjoint(splits)
    digests = json.dumps({k: v["digest"] for k, v in summary["splits"].items()}, sort_keys=True)
    summary["digest"] = hashlib.sha256(digests.encode("utf-8")).hexdigest()
    return summary
