"""
mergeslide/prompts.py
Class-aware prompt definitions for the configured WSI task sequence.

Each task function returns (class_prompts, templates) where:
  - class_prompts: List[List[str]] — one list of synonym prompts per class
  - templates: List[str] — shared prompt templates using 'CLASSNAME' placeholder
"""

# Shared H&E histopathology templates used by all tasks.
TEMPLATES = [
    "CLASSNAME.",
    "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.",
    "an image showing CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E.",
]


# def camelyon17_prompts():
#     """CAMELYON17: Lymph node metastasis burden."""
#     prompts = [
#         [
#             'negative lymph node',
#             'lymph node without tumor metastasis',
#             'no metastatic breast carcinoma in lymph node',
#         ],
#         [
#             'isolated tumor cells in lymph node',
#             'isolated tumor cell metastasis',
#             'very small cluster of metastatic tumor cells in lymph node',
#         ],
#         [
#             'micrometastasis in lymph node',
#             'small metastatic breast carcinoma focus in lymph node',
#             'microscopic metastatic tumor deposit in lymph node',
#         ],
#         [
#             'macrometastasis in lymph node',
#             'large metastatic breast carcinoma deposit in lymph node',
#             'overt metastatic tumor deposit in lymph node',
#         ],
#     ]
#     return prompts, TEMPLATES
def camelyon17_prompts():
    prompts = [
        [
            "negative sentinel lymph node",
            "sentinel lymph node without metastatic carcinoma",
            "no tumor cells in breast cancer sentinel lymph node",
            "benign lymph node tissue without metastasis",
        ],
        [
            "isolated tumor cells in sentinel lymph node",
            "single tumor cells or tiny clusters in lymph node",
            "very small isolated tumor cell cluster less than 0.2 mm",
            "isolated tumor cells not counted as lymph node metastasis",
        ],
        [
            "lymph node micrometastasis",
            "small metastatic breast carcinoma deposit in sentinel lymph node",
            "micrometastatic tumor deposit between 0.2 mm and 2.0 mm",
            "microscopic breast cancer metastasis in lymph node",
        ],
        [
            "lymph node macrometastasis",
            "large metastatic breast carcinoma deposit in sentinel lymph node",
            "macrometastatic tumor deposit greater than 2.0 mm",
            "overt breast cancer metastasis in lymph node",
        ],
    ]
    return prompts, TEMPLATES

def brca_prompts():
    """TCGA-BRCA: Invasive Ductal vs. Invasive Lobular Carcinoma."""
    prompts = [
        [
            'invasive ductal carcinoma',
            'breast invasive ductal carcinoma',
            'invasive ductal carcinoma of the breast',
            'invasive carcinoma of the breast, ductal pattern',
            'idc',
        ],
        [
            'invasive lobular carcinoma',
            'breast invasive lobular carcinoma',
            'invasive lobular carcinoma of the breast',
            'invasive carcinoma of the breast, lobular pattern',
            'ilc',
        ],
    ]
    return prompts, TEMPLATES


def nsclc_prompts():
    """TCGA-NSCLC: Lung Adenocarcinoma vs. Squamous Cell Carcinoma."""
    prompts = [
        [
            'adenocarcinoma',
            'lung adenocarcinoma',
            'adenocarcinoma of the lung',
            'luad',
        ],
        [
            'squamous cell carcinoma',
            'lung squamous cell carcinoma',
            'squamous cell carcinoma of the lung',
            'lusc',
        ],
    ]
    return prompts, TEMPLATES


def rcc_prompts():
    """TCGA-RCC: Clear Cell / Papillary / Chromophobe Renal Cell Carcinoma."""
    prompts = [
        [
            'clear cell renal cell carcinoma',
            'renal cell carcinoma, clear cell type',
            'renal cell carcinoma of the clear cell type',
            'clear cell rcc',
        ],
        [
            'papillary renal cell carcinoma',
            'renal cell carcinoma, papillary type',
            'renal cell carcinoma of the papillary type',
            'papillary rcc',
        ],
        [
            'chromophobe renal cell carcinoma',
            'renal cell carcinoma, chromophobe type',
            'renal cell carcinoma of the chromophobe type',
            'chromophobe rcc',
            'chromophobe renal cell carcinoma, which is a rare type of kidney cancer that forms in the cells lining the small tubules in the kidney. These small tubules help filter waste from the blood, making urine.',
        ],
    ]
    return prompts, TEMPLATES


def esca_prompts():
    """TCGA-ESCA: Esophageal Adenocarcinoma vs. Squamous Cell Carcinoma."""
    prompts = [
        [
            'adenocarcinoma',
            'esophageal adenocarcinoma',
            'adenocarcinoma of the esophagus',
            'esad',
        ],
        [
            'squamous cell carcinoma',
            'esophageal squamous cell carcinoma',
            'squamous cell carcinoma of the esophagus',
            'essc',
        ],
    ]
    return prompts, TEMPLATES


def tgct_prompts():
    """TCGA-TGCT: Seminoma vs. Mixed Germ Cell Tumor."""
    prompts = [
        [
            'seminoma',
            'testicular seminoma',
            'seminoma of the testis',
        ],
        [
            'mixed germ cell tumor',
            'testicular mixed germ cell tumor',
            'mixed germ cell tumor of the testis',
        ],
    ]
    return prompts, TEMPLATES


def cesc_prompts():
    """TCGA-CESC: Cervical Adenocarcinoma vs. Squamous Cell Carcinoma."""
    prompts = [
        [
            'adenocarcinoma',
            'cervical adenocarcinoma',
            'adenocarcinoma of the cervix uteri',
        ],
        [
            'squamous cell carcinoma',
            'cervical squamous cell carcinoma',
            'squamous cell carcinoma of the cervix uteri',
        ],
    ]
    return prompts, TEMPLATES


# def bracs_prompts():
#     """BRACS: Benign / Atypical / Malignant breast lesions."""
#     prompts = [
#         [
#             'benign breast lesion',
#             'benign breast tissue',
#             'non-malignant breast lesion',
#             'benign breast pathology',
#         ],
#         [
#             'atypical breast lesion',
#             'breast lesion with epithelial atypia',
#             'atypical breast tissue',
#             'borderline breast lesion',
#         ],
#         [
#             'malignant breast tumor',
#             'breast carcinoma',
#             'malignant breast lesion',
#             'invasive or in situ breast carcinoma',
#         ],
#     ]
#     return prompts, TEMPLATES

# def herohe_prompts():
#     """HEROHE: HER2 status negative vs positive."""
#     prompts = [
#         [
#             'HER2 negative breast cancer',
#             'breast cancer with negative HER2 status',
#             'HER2 non-amplified breast tumor',
#             'HER2-negative invasive breast carcinoma',
#         ],
#         [
#             'HER2 positive breast cancer',
#             'breast cancer with positive HER2 status',
#             'HER2 amplified breast tumor',
#             'HER2-positive invasive breast carcinoma',
#         ],
#     ]
#     return prompts, TEMPLATES


# def ubc_ocean_prompts():
#     """UBC-OCEAN: Ovarian carcinoma histologic subtypes."""
#     prompts = [
#         [
#             'high grade serous carcinoma',
#             'ovarian high grade serous carcinoma',
#             'high grade serous ovarian carcinoma',
#             'HGSC ovarian carcinoma',
#         ],
#         [
#             'endometrioid carcinoma',
#             'ovarian endometrioid carcinoma',
#             'endometrioid ovarian carcinoma',
#         ],
#         [
#             'clear cell carcinoma',
#             'ovarian clear cell carcinoma',
#             'clear cell ovarian carcinoma',
#         ],
#         [
#             'low grade serous carcinoma',
#             'ovarian low grade serous carcinoma',
#             'low grade serous ovarian carcinoma',
#             'LGSC ovarian carcinoma',
#         ],
#         [
#             'mucinous carcinoma',
#             'ovarian mucinous carcinoma',
#             'mucinous ovarian carcinoma',
#         ],
#     ]
#     return prompts, TEMPLATES
def bracs_prompts():
    prompts = [
        [
            "benign breast lesion in BRACS histology",
            "normal or benign breast tissue",
            "pathological benign breast lesion",
            "usual ductal hyperplasia or benign breast change",
        ],
        [
            "atypical breast lesion in BRACS histology",
            "flat epithelial atypia or atypical ductal hyperplasia",
            "breast epithelial atypia without invasive carcinoma",
            "premalignant atypical breast lesion",
        ],
        [
            "malignant breast lesion in BRACS histology",
            "ductal carcinoma in situ or invasive breast carcinoma",
            "breast carcinoma lesion",
            "malignant epithelial breast tumor",
        ],
    ]
    return prompts, TEMPLATES

# def herohe_prompts():
#     prompts = [
#         [
#             "HER2 negative invasive breast cancer",
#             "breast carcinoma with absent HER2 overexpression",
#             "HER2 non-amplified breast carcinoma",
#             "invasive breast tumor with negative HER2 receptor status",
#         ],
#         [
#             "HER2 positive invasive breast cancer",
#             "breast carcinoma with HER2 overexpression",
#             "HER2 amplified breast carcinoma",
#             "invasive breast tumor with positive HER2 receptor status",
#         ],
#     ]
#     return prompts, TEMPLATES

def herohe_prompts():
    prompts = [
        [
            "HER2-negative",
            "HER2 negative invasive breast cancer",
            "breast carcinoma with absent HER2 overexpression",
            "HER2 non-amplified breast carcinoma",
            "invasive breast tumor with negative HER2 receptor status",

        ],
        [
            "HER2-positive",
            "HER2 positive invasive breast cancer",
            "breast carcinoma with HER2 overexpression",
            "HER2 amplified breast carcinoma",
            "invasive breast tumor with positive HER2 receptor status",
        ],
    ]
    return prompts, TEMPLATES


def ubc_ocean_prompts():
    prompts = [
        [
            "ovarian high grade serous carcinoma",
            "high grade serous ovarian carcinoma",
            "HGSC ovarian carcinoma",
            "high grade serous carcinoma of the ovary",
        ],
        [
            "ovarian endometrioid carcinoma",
            "endometrioid ovarian carcinoma",
            "endometrioid carcinoma of the ovary",
        ],
        [
            "ovarian clear cell carcinoma",
            "clear cell ovarian carcinoma",
            "clear cell carcinoma of the ovary",
        ],
        [
            "ovarian low grade serous carcinoma",
            "low grade serous ovarian carcinoma",
            "LGSC ovarian carcinoma",
            "low grade serous carcinoma of the ovary",
        ],
        [
            "ovarian mucinous carcinoma",
            "mucinous ovarian carcinoma",
            "mucinous carcinoma of the ovary",
        ],
    ]
    return prompts, TEMPLATES
# Ordered list of all task prompt functions:
# Camelyon17 -> 6 TCGA -> BRACS -> HEROHE -> UBC-OCEAN
ALL_TASK_PROMPTS = [
    camelyon17_prompts,
    brca_prompts,
    rcc_prompts,
    nsclc_prompts,
    esca_prompts,
    tgct_prompts,
    cesc_prompts,
    bracs_prompts,
    herohe_prompts,
    ubc_ocean_prompts,
]
