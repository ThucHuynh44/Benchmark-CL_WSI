"""
mergeslide/prompts.py
Class-aware prompt definitions for all six TCGA tasks.

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


# Ordered list of all task prompt functions (matches task_id 0–5)
ALL_TASK_PROMPTS = [brca_prompts, rcc_prompts, nsclc_prompts, esca_prompts, tgct_prompts, cesc_prompts]
