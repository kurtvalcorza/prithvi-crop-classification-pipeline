"""Labelled-chip dataset contract for adapting the crop-classification model: the pinned multi-temporal crop
sample, role assignment by spatial block, BYOD loaders and sample export.

The default dataset is **real**: 60 labelled 224 × 224 chips of the HLS multi-temporal crop classification dataset
(NASA IMPACT / IBM, CC BY 4.0) — three HLS dates of six bands over the contiguous United States in 2022, with a
13-class label derived from the USDA Cropland Data Layer — drawn on 2026-09-20 from the 218 chips of the
`validation_chips.tgz` archive that carry both an image and a mask (the archive lists 397 images and 390 masks
that pair up for only 218 chip ids). Every one of the 60 chips was part of the upstream *validation* split, i.e.
the split the published checkpoint was selected on (`best_mIoU_epoch_80`), so the frozen model has seen these
chips as validation data but was never trained on them. Roles are assigned per 4 × 4-chip block of the chip grid
(a seeded hash of the block; 36 train / 12 validation / 12 test, each stratified by dominant class so all 13
classes occur in every role) — chips of one block never straddle roles, but neighbouring blocks may, so the
split is by block, not by region; real data must be split by region. The dataset is distributed as one 1.18 GB
gzipped tarball on the Hugging Face Hub; the tarball is pinned by byte size and SHA-256, each pinned member is
pinned again by size and SHA-256 and extracted **without** `extractall` into the cache, and everything else in
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
TAR_SHA256 = "59407373b38c575a081cfbf1d70c92cfc45430be5d5e2f6a51c4b83a642a23fb"
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
        "chip_115_502",
        "train",
        "validation_chips/chip_115_502_merged.tif",
        1808174,
        "6171327f361c13644932156c0a2925e10441b9505aebc5c25af0f40188079b75",
        "validation_chips/chip_115_502.mask.tif",
        52124,
        "3971ce9e0ec74bd9be261c78ce1a8d796034c766fe0e8c43c143cd589cb34738",
    ),
    (
        "chip_122_504",
        "train",
        "validation_chips/chip_122_504_merged.tif",
        1808174,
        "515a547366bb8be6159cefae5dcae1f932bf089b1ed4857e51009bbd29303200",
        "validation_chips/chip_122_504.mask.tif",
        52124,
        "682d116b3d86e092de504477e371fcbff062ad0f4a1f12806ccc9c5b3f365e04",
    ),
    (
        "chip_149_478",
        "train",
        "validation_chips/chip_149_478_merged.tif",
        1808174,
        "d4f5dd028880329c6639c15944eff9183090082d5837e52fd6e8755184593ce4",
        "validation_chips/chip_149_478.mask.tif",
        52124,
        "2c8463d1d23f8849ddf831b12d4720755c9408bd90d3f8905089bf1b8dd2012f",
    ),
    (
        "chip_164_262",
        "train",
        "validation_chips/chip_164_262_merged.tif",
        1808174,
        "0a00323ed6ef786c2fc38cbbdc0746a2fa9d681a451743468b0c44401730fd95",
        "validation_chips/chip_164_262.mask.tif",
        52124,
        "a45b51e4d670270efe5e8cc9d3a7a426b1272c6e9282d9161e0ccb8882444498",
    ),
    (
        "chip_164_444",
        "train",
        "validation_chips/chip_164_444_merged.tif",
        1808174,
        "0ec2b7e93f98d498e7309ac0a1a3e4d39670b6391d88f61c3a6c0590dda619fe",
        "validation_chips/chip_164_444.mask.tif",
        52124,
        "c8ef112d23464097d29cd833e64416fa11e0bbedc017b57b1676817a638481b2",
    ),
    (
        "chip_164_519",
        "train",
        "validation_chips/chip_164_519_merged.tif",
        1808174,
        "e567a6df9e92f97d60586168d4250f98574571cef0b0f67c7514030eb0400245",
        "validation_chips/chip_164_519.mask.tif",
        52124,
        "f4a5a6ef61838683346fe5a5274093767987d09fe287322144252725a7a24269",
    ),
    (
        "chip_165_601",
        "train",
        "validation_chips/chip_165_601_merged.tif",
        1808174,
        "142b240be3e65d646f3c05ae1cfd1e1677f09202c0633d32e099dd70b3189d34",
        "validation_chips/chip_165_601.mask.tif",
        52124,
        "2036b4b13b0994a5f68dc8b75a7602f1478de990ce47faeedb2a04327110bcdb",
    ),
    (
        "chip_168_268",
        "train",
        "validation_chips/chip_168_268_merged.tif",
        1808174,
        "f95b9716915572f0cf45f819c5ecf66218c291936f5c6e989eb1164358a9d0cd",
        "validation_chips/chip_168_268.mask.tif",
        52124,
        "09255e8f15675458e5015f7fd84396b6d4ba3b15ce8c074ad7a777c1a55bd52c",
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
        "chip_179_609",
        "train",
        "validation_chips/chip_179_609_merged.tif",
        1808174,
        "9c4b14b9a17059587178e3456f08ed6c6de39ff1e366cf6d02ae4f4385dcdc07",
        "validation_chips/chip_179_609.mask.tif",
        52124,
        "9967199887e9da03b8f3d14f077a02ae23d85b294c194fc3d9642f62eb59c758",
    ),
    (
        "chip_181_238",
        "train",
        "validation_chips/chip_181_238_merged.tif",
        1808174,
        "31e929d112d2a2e7d3f5ba6d05945d84a974146bf9dc1349f8f9ecffd35f6cc3",
        "validation_chips/chip_181_238.mask.tif",
        52124,
        "111c418be9a7087cbe0fb493bb0ed56ced285f80ebd15a95ce0eed0937180e49",
    ),
    (
        "chip_194_468",
        "train",
        "validation_chips/chip_194_468_merged.tif",
        1808174,
        "1718ec93475feb739ce1d43013f734d181810135dbeeaa83a74c54cf3b929fe0",
        "validation_chips/chip_194_468.mask.tif",
        52124,
        "4c04813b5b7f7ec6baa836becc1b26367020457fc3847e12a76a50bb1e36cc12",
    ),
    (
        "chip_201_445",
        "train",
        "validation_chips/chip_201_445_merged.tif",
        1808174,
        "624e6f0fff598f54da587b08780a20aa00e24d5861549e19a0ee14724be5cf35",
        "validation_chips/chip_201_445.mask.tif",
        52124,
        "099867a730005631656ac33b21e61de2e5eef6dd254c6266f756bc23e7f876c2",
    ),
    (
        "chip_204_480",
        "train",
        "validation_chips/chip_204_480_merged.tif",
        1808174,
        "874138712932bfda7d01b0488134bb0bb1be5eb9854b93c8285e26c73c680f68",
        "validation_chips/chip_204_480.mask.tif",
        52124,
        "df8351711eef9504befaaa06e4f515a164c3b3ed8665e18fa6cf4f802c4f0026",
    ),
    (
        "chip_213_293",
        "train",
        "validation_chips/chip_213_293_merged.tif",
        1808174,
        "fd1ea6dbca10dbc37fd77e771728edbf6117416143ffbe22006585e11545abce",
        "validation_chips/chip_213_293.mask.tif",
        52124,
        "5c28386e3397192eb06bd232761e3f4fc98db39846dc210f57d18542ba8bbcbb",
    ),
    (
        "chip_215_425",
        "train",
        "validation_chips/chip_215_425_merged.tif",
        1808174,
        "a97ed04ef6242d6872b3a22e57aa01e5b71e2f8b606de53bcdf5a184d3ceccd2",
        "validation_chips/chip_215_425.mask.tif",
        52124,
        "92e3ccb954e5303dc7fdb4a2d65b3e6b1ecd0a4226949090513f0bffd51a4eed",
    ),
    (
        "chip_220_308",
        "train",
        "validation_chips/chip_220_308_merged.tif",
        1808174,
        "83865078589af91acae9410805d713359a6fcd38c0fa0cc669325387f732acb9",
        "validation_chips/chip_220_308.mask.tif",
        52124,
        "3dda3aa633a3bd60b2107f20a076c599ba1df6a644f30d88ac15f279a32511aa",
    ),
    (
        "chip_228_297",
        "train",
        "validation_chips/chip_228_297_merged.tif",
        1808174,
        "79a3264843d1f249ec6e2d739c623098b43a21c09481989c74e0e54f6fe028b3",
        "validation_chips/chip_228_297.mask.tif",
        52124,
        "b722f51c0380636985950a5ba83ff4cffa5c24b367597b3e1f38d94a32148d0a",
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
        "chip_232_595",
        "train",
        "validation_chips/chip_232_595_merged.tif",
        1808174,
        "a212d100ea06594b40ae2cbd66b7324200937124195f594c1c76892ee3bac835",
        "validation_chips/chip_232_595.mask.tif",
        52124,
        "64180500243fda6678e375ce305a3e9519af42911c534c7b4d3ba7719d8cadd0",
    ),
    (
        "chip_244_267",
        "train",
        "validation_chips/chip_244_267_merged.tif",
        1808174,
        "84ba53c97e4728ba5f19f863f8119ea709aa7f33d2376b35b197dad790e977f1",
        "validation_chips/chip_244_267.mask.tif",
        52124,
        "5992c23423c7f69e148f36d9057b2a7361d6b1cacfb4785846cc4cacf08e1c56",
    ),
    (
        "chip_244_595",
        "train",
        "validation_chips/chip_244_595_merged.tif",
        1808174,
        "7c1088ebd83e4137ee53a006fa0a7bba3ec89dd75f9da4c0a09a8bae281619a3",
        "validation_chips/chip_244_595.mask.tif",
        52124,
        "8d9a9970307e2def3e74903c814a15dbc1b74a35d98716d587b20922ad76418b",
    ),
    (
        "chip_245_595",
        "train",
        "validation_chips/chip_245_595_merged.tif",
        1808174,
        "5337756688122692e2ae54b080e59f3188594dc66d469c3f5ad6289d949753e7",
        "validation_chips/chip_245_595.mask.tif",
        52124,
        "698701e17e7b03e209408965cd5adcf5c75fb5c106ba8f44c60f1464919e8bb2",
    ),
    (
        "chip_251_589",
        "train",
        "validation_chips/chip_251_589_merged.tif",
        1808174,
        "c72f088356b80f3fd4f1d2398bfd2ba78e9f4e9ea809f7458fb71d661f924a76",
        "validation_chips/chip_251_589.mask.tif",
        52124,
        "30641091a9c7b70d055bbe48fda2dea07b78fc7770aee7952e25e6a8636d7c46",
    ),
    (
        "chip_256_267",
        "train",
        "validation_chips/chip_256_267_merged.tif",
        1808174,
        "65dc456357e7fbd65d40c0181077bca8f88c94fe51430c76a950e2d403ff70b9",
        "validation_chips/chip_256_267.mask.tif",
        52124,
        "49439cf6f1b866b3fab4006cf0e2efeccdaecbc150ca2c70917e066306d6c0b8",
    ),
    (
        "chip_269_263",
        "train",
        "validation_chips/chip_269_263_merged.tif",
        1808174,
        "bf63854cf77ce05e8b5c1f103872452265c3e717750eded5545802d115d496d1",
        "validation_chips/chip_269_263.mask.tif",
        52124,
        "e6efc140775ab2ccb411d81ac46ffccc92c63efaee42db68e983017ed3211169",
    ),
    (
        "chip_275_475",
        "train",
        "validation_chips/chip_275_475_merged.tif",
        1808174,
        "410c5f2bb96ead47af05eca50e295b1167d1a8400235d4363f717d0a1b6a807b",
        "validation_chips/chip_275_475.mask.tif",
        52124,
        "2d0d2bb57248e8416fb27ae6c52f74479bcffef9734b36d78b163ecee7feda4e",
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
        "chip_292_542",
        "train",
        "validation_chips/chip_292_542_merged.tif",
        1808174,
        "37e1ec9a9d1c326cca25ef11a2309e1d7b0b2a7a01b0ab20b7acc77e131735dc",
        "validation_chips/chip_292_542.mask.tif",
        52124,
        "3b167cb1f5ebc0e0eaffe0cbdada850cf9464c2ddb3ca7761551ca3fe479c578",
    ),
    (
        "chip_304_299",
        "train",
        "validation_chips/chip_304_299_merged.tif",
        1808174,
        "189375550af4fc64480cbb831d57b84bff9cf342330e5ca204c8b9700fbf678b",
        "validation_chips/chip_304_299.mask.tif",
        52124,
        "07978762e719dc33e55428dba4aacb00b479df58b5fbe43cf1151bd6f0a485d2",
    ),
    (
        "chip_314_518",
        "train",
        "validation_chips/chip_314_518_merged.tif",
        1808174,
        "f079380bcf25e5e3bf70a9664bb6e47ed141d788f5ed80d1dcdc7531c563d593",
        "validation_chips/chip_314_518.mask.tif",
        52124,
        "25799706afd301d786e5e6e606c898531eb797b6d1f6ed66af82edbdea1c95f6",
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
        "chip_094_321",
        "validation",
        "validation_chips/chip_094_321_merged.tif",
        1808174,
        "0d19c7f185d1b8878dd467557a75c554d4e2547a2b860a06a5cc32de0463f9cb",
        "validation_chips/chip_094_321.mask.tif",
        52124,
        "49240158ee0afd784b2f4c324fadf597ec61b9847c4a41ffb3041f74eef399d0",
    ),
    (
        "chip_124_301",
        "validation",
        "validation_chips/chip_124_301_merged.tif",
        1808174,
        "402ba0f3055484b269b6d19464c131f2f22a026702d9a00e1f86d4c8abb447af",
        "validation_chips/chip_124_301.mask.tif",
        52124,
        "c2d7a6cd6db4be75c3c5185794e6f95dd0b286ef9c7973d8df7d68b7064f16cd",
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
        "chip_141_401",
        "validation",
        "validation_chips/chip_141_401_merged.tif",
        1808174,
        "966e68bf13619ffd9f5894e02e0f67970d3d1b5c1a2bc49f405e826a2876db4b",
        "validation_chips/chip_141_401.mask.tif",
        52124,
        "b60785b49d4fe37fda4f726ea4b9f78ca8fe14b21593c1da3dcb067e4e743606",
    ),
    (
        "chip_152_477",
        "validation",
        "validation_chips/chip_152_477_merged.tif",
        1808174,
        "87e9192c02360bf5b43404b58602143005dd27b0ed4b460ca8f6ba3ff81eb75c",
        "validation_chips/chip_152_477.mask.tif",
        52124,
        "ae679128aea968eb5f72e8328cc6d0ee2b2c706310b9701a995fbc761fe72325",
    ),
    (
        "chip_183_502",
        "validation",
        "validation_chips/chip_183_502_merged.tif",
        1808174,
        "d168b1a1dbfe0d6ba5df9234a4684da6a5bd74aa9829d5c7431d48808e8343e3",
        "validation_chips/chip_183_502.mask.tif",
        52124,
        "0da04f0e5f369bb5495b366afa5769e4f04862bcaf6b4485e1ba7a8bcc681ec2",
    ),
    (
        "chip_215_454",
        "validation",
        "validation_chips/chip_215_454_merged.tif",
        1808174,
        "895b529a896aa00221bd6b69a6f2fab8057163e2aeb88bbfde0becbc9ac31f32",
        "validation_chips/chip_215_454.mask.tif",
        52124,
        "a903d2e358bfd24eaf52edfc248478f8e51edef1ec7b6b6fce263cac93ada953",
    ),
    (
        "chip_235_446",
        "validation",
        "validation_chips/chip_235_446_merged.tif",
        1808174,
        "84692d0d89a0f38fceee876b0db8c5ece5e4d6162e135a50d3880f033317515b",
        "validation_chips/chip_235_446.mask.tif",
        52124,
        "35a2856b12d391777d7f6714d53ba04c457b38f48a57bca2428aa2d8faa22917",
    ),
    (
        "chip_255_273",
        "validation",
        "validation_chips/chip_255_273_merged.tif",
        1808174,
        "995f8135ad5cbc51046fdb387c89c1725fbc59db067f3a5a02dafc2f93592865",
        "validation_chips/chip_255_273.mask.tif",
        52124,
        "e93d3fe397d1dc53ff5c3baace5c2a2195e6df9abbe3bf11923d12463577414f",
    ),
    (
        "chip_271_264",
        "validation",
        "validation_chips/chip_271_264_merged.tif",
        1808174,
        "4427829fd16bd8b88fc902e8f7b5c9bfd0ed9516f5e662adbc09b6c47a410c90",
        "validation_chips/chip_271_264.mask.tif",
        52124,
        "3ccbc4ff357115fb27a6b46c9e9227c73cb988b685a687c737fb992f3f7cb05e",
    ),
    (
        "chip_330_498",
        "validation",
        "validation_chips/chip_330_498_merged.tif",
        1808174,
        "05631071a012bbb44a036827f6b336cead59ee0ecb36f9d430d504273044bf9b",
        "validation_chips/chip_330_498.mask.tif",
        52124,
        "6af39dcf09e44dd7c9a5aece526730ee274af1cc03c1eae4fa397c11a2236545",
    ),
    (
        "chip_097_358",
        "test",
        "validation_chips/chip_097_358_merged.tif",
        1808174,
        "f89479add462408cd782f708b85c4500066ab22e178d8638af629aac0864774b",
        "validation_chips/chip_097_358.mask.tif",
        52124,
        "2edea29ddd2bbb4cd0671212df12118c0e042509fb3e1f7b70af3ff696674035",
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
        "chip_110_421",
        "test",
        "validation_chips/chip_110_421_merged.tif",
        1808174,
        "e5665bb06f3489de5f5fce7183393b13af21692f236ced4724a09cf5131ff929",
        "validation_chips/chip_110_421.mask.tif",
        52124,
        "b69cfcf032158695b3490972e69078eed766a440369b4c653394d2fc0d9b4fda",
    ),
    (
        "chip_121_436",
        "test",
        "validation_chips/chip_121_436_merged.tif",
        1808174,
        "32994f480af20970d7c8ce7208e2d9bde0d7c23fc76df14b87ce851a0e8b7e64",
        "validation_chips/chip_121_436.mask.tif",
        52124,
        "2e8461cccdfe4a096ec28689e59d4cdf3c6df21bcfd4ddc50122323b5b5f5c83",
    ),
    (
        "chip_124_439",
        "test",
        "validation_chips/chip_124_439_merged.tif",
        1808174,
        "238f58b8fc2d8ec9e712d924b61596695623c37575aa542424bf3462888da198",
        "validation_chips/chip_124_439.mask.tif",
        52124,
        "9a1cbc7c862e939535c075ab3e46a549398286c026da106d510fe1970554fdcd",
    ),
    (
        "chip_131_422",
        "test",
        "validation_chips/chip_131_422_merged.tif",
        1808174,
        "f2bc5e2deeb338706da32ee490837bce85125f4ba268c870b592924874781b86",
        "validation_chips/chip_131_422.mask.tif",
        52124,
        "cfeea7aa990b732e94a013f898663fbefbda0d93a7301481a16288481d04919c",
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
        "chip_189_360",
        "test",
        "validation_chips/chip_189_360_merged.tif",
        1808174,
        "a9ea2470de560c7b55bc04c66e7b091f79fb97a8a7ba5fa15acb6e0363d12c9f",
        "validation_chips/chip_189_360.mask.tif",
        52124,
        "62c0d410acd2a73e7d6d58b39b2803d071f55d2bb35fad1ee195a3018461564c",
    ),
    (
        "chip_201_433",
        "test",
        "validation_chips/chip_201_433_merged.tif",
        1808174,
        "533b472ab8fe3a9e8a24d578a65d177a0f504c932e09905d43cff58f7942c384",
        "validation_chips/chip_201_433.mask.tif",
        52124,
        "7692b6c130f50f583c5a577197214e6333462737c7233577b3d8ab5a1ecdc33d",
    ),
    (
        "chip_223_438",
        "test",
        "validation_chips/chip_223_438_merged.tif",
        1808174,
        "ef27699bdd853da6d3bc56a0c41227f3da25b816491408b37aa75b1805c08e74",
        "validation_chips/chip_223_438.mask.tif",
        52124,
        "db4de7fa2ba72be5feec5c297529dacdb3fd076802f53c26a55b2fa28ada4746",
    ),
    (
        "chip_234_289",
        "test",
        "validation_chips/chip_234_289_merged.tif",
        1808174,
        "6dae19d25f8b24306ac5ec6b1cf6e49fb8567766b972dc179dd0f24023cff39d",
        "validation_chips/chip_234_289.mask.tif",
        52124,
        "b9e589d1245c6d07ac8aba2b488f0138dae0f97a202b92eac8797a2fa1bfe325",
    ),
    (
        "chip_304_528",
        "test",
        "validation_chips/chip_304_528_merged.tif",
        1808174,
        "8455216599a37737cc668b7292093c7051ed613e6e184007d14343f39c4216ba",
        "validation_chips/chip_304_528.mask.tif",
        52124,
        "dcd33d3456ce0fe713e8e3730fb5d5c1e8b80a92735a4734a7cc91b2524d958a",
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
