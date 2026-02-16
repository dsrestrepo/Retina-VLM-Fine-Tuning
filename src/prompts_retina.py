import pandas as pd

def convert_eye(eye):
    return {1: "right", 2: "left"}.get(eye, "no eye reported")

def convert_condition(val):
    # conversion for binary conditions
    try:
        val_str = str(val).lower()
        if val_str in ['1', 'yes', 'true']: return "present"
        if val_str in ['0', 'no', 'false']: return "absent"
    except:
        pass
    return "absent"

def convert_anatomical(val):
    return {1: "normal", 2: "abnormal"}.get(val, "unknown")

def convert_sex_brset(val):
    # BRSET sex conversion
    # Assuming '1' or 'Male' / '2' or 'Female'
    val_str = str(val).lower()
    if val_str in ['1', 'male', 'm', 'masculino']: return "male"
    if val_str in ['2', 'female', 'f', 'feminino', '0']: return "female"
    return "sex not reported"

def convert_sex_mbrset(val):
    # mBRSET sex conversion: 0=Female, 1=Male (from mbrset.ipynb)
    try:
        if int(val) == 1: return "male"
        if int(val) == 0: return "female"
    except:
        pass
    return "sex not reported"

def binary_to_text(val, text_true, text_false):
    try:
        if int(val) == 1: return text_true
    except:
        if str(val).lower() in ['yes', 'true', '1']: return text_true
    return text_false

education_map = {
    1.0: "illiterate",
    2.0: "with incomplete primary education",
    3.0: "with complete primary education",
    4.0: "with incomplete secondary education",
    5.0: "with complete secondary education",
    6.0: "with incomplete tertiary education",
    7.0: "with complete tertiary education"
}


#########################################################################
####################### BRSET and mBRSET prompts ########################
#########################################################################


######################## Image and Text Prompts #########################
def BRSET_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {float(str(row['patient_age']).replace('O', '0').replace(',', '.'))} years"
        if not pd.isnull(row['patient_age'])
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {float(str(row['diabetes_time_y']).replace('O', '0').replace(',', '.'))} years"
        if not pd.isnull(row['diabetes_time_y']) and row['diabetes_time_y'] != 'Não'
        else "with no reported diabetes duration"
    )

    # Comorbidities
    comorb_phrase = (
        "with no comorbidities reported"
        if pd.isnull(row['comorbidities'])
        else f"with comorbidities: {row['comorbidities']}"
    )

    # Insulin use
    insulin_phrase = (
        "using insulin" if str(row["insuline"]).strip().lower() == "yes" else "not using insulin"
    )

    # Anatomical description
    anatomy = (
        f"The optic disc appears {convert_anatomical(row['optic_disc'])}, "
        f"the vessels are {convert_anatomical(row['vessels'])}, "
        f"and the macula is {convert_anatomical(row['macula'])}."
    )

    # Disease/condition labels
    condition_fields = [
        "macular_edema", "scar", "nevus", "amd", "vascular_occlusion",
        "hypertensive_retinopathy", "drusens", "hemorrhage", "retinal_detachment",
        "myopic_fundus", "increased_cup_disc", "other"
    ]

    conditions = ", ".join(
        f"{field.replace('_', ' ')}: {convert_condition(row[field])}"
        for field in condition_fields
    )

    # Compose the full description
    description = (
        f"A {convert_sex_brset(row['patient_sex'])} patient {age_phrase}, "
        f"{diabetes_phrase}, {insulin_phrase}, and {comorb_phrase}. "
        f"{anatomy} Conditions include: {conditions}."
    )

    # Compose the final prompt
    return f"""
{description}

Based on the provided patient information and the associated fundus image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()


def mBRSET_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {row['age']} years" 
        if not pd.isnull(row['age']) 
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {row['dm_time']} years" 
        if not pd.isnull(row['dm_time']) 
        else "with no reported diabetes duration"
    )

    # Educational level
    education = education_map.get(row['educational_level'], "with no educational level reported")

    # Build descriptions
    sex = convert_sex_mbrset(row['sex'])
    insulin = binary_to_text(row['insulin'], "using insulin", "not using insulin")
    oral = binary_to_text(row['oraltreatment_dm'], "on oral treatment for diabetes", "not on oral treatment for diabetes")
    hypertension = binary_to_text(row['systemic_hypertension'], "with systemic hypertension", "without systemic hypertension")
    alcohol = binary_to_text(row['alcohol_consumption'], "consumes alcohol", "does not consume alcohol")
    smoking = binary_to_text(row['smoking'], "smokes", "does not smoke")
    obesity = binary_to_text(row['obesity'], "with obesity", "without obesity")
    vascular = binary_to_text(row['vascular_disease'], "has vascular disease", "does not have vascular disease")
    infarction = binary_to_text(row['acute_myocardial_infarction'], "has a history of acute myocardial infarction", "no history of acute myocardial infarction")
    nephropathy = binary_to_text(row['nephropathy'], "with nephropathy", "without nephropathy")
    neuropathy = binary_to_text(row['neuropathy'], "with neuropathy", "without neuropathy")
    diabetic_foot = binary_to_text(row['diabetic_foot'], "has diabetic foot", "does not have diabetic foot")

    # Compose patient description
    description = (
        f"A {sex} patient {age_phrase}, {diabetes_phrase}, {insulin}, and {oral}. "
        f"The patient is {hypertension}, {alcohol}, {smoking}, {obesity}, and {vascular}. "
        f"Medical history includes: {infarction}, {nephropathy}, {neuropathy}, and {diabetic_foot}. "
        f"The patient is {education}."
    )

    # LLM prompt
    return f"""
{description}

Based on the provided patient information and the associated fundus image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()



######################## Only Image Prompts #########################

BRSET_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
"""

mBRSET_ONLY_IMAGE_TEXT_PROMPT = f"""
Based on the image, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
"""


######################## Only Text Prompts #########################

# Define the text prompt for BRSET.
def BRSET_ONLY_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {float(str(row['patient_age']).replace('O', '0').replace(',', '.'))} years"
        if not pd.isnull(row['patient_age'])
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {float(str(row['diabetes_time_y']).replace('O', '0').replace(',', '.'))} years"
        if not pd.isnull(row['diabetes_time_y']) and row['diabetes_time_y'] != 'Não'
        else "with no reported diabetes duration"
    )

    # Comorbidities
    comorb_phrase = (
        "with no comorbidities reported"
        if pd.isnull(row['comorbidities'])
        else f"with comorbidities: {row['comorbidities']}"
    )

    # Insulin use
    insulin_phrase = (
        "using insulin" if str(row["insuline"]).strip().lower() == "yes" else "not using insulin"
    )

    # Anatomical description
    anatomy = (
        f"The optic disc appears {convert_anatomical(row['optic_disc'])}, "
        f"the vessels are {convert_anatomical(row['vessels'])}, "
        f"and the macula is {convert_anatomical(row['macula'])}."
    )

    # Disease/condition labels
    condition_fields = [
        "macular_edema", "scar", "nevus", "amd", "vascular_occlusion",
        "hypertensive_retinopathy", "drusens", "hemorrhage", "retinal_detachment",
        "myopic_fundus", "increased_cup_disc", "other"
    ]

    conditions = ", ".join(
        f"{field.replace('_', ' ')}: {convert_condition(row[field])}"
        for field in condition_fields
    )

    # Compose the full description
    description = (
        f"A {convert_sex_brset(row['patient_sex'])} patient {age_phrase}, "
        f"{diabetes_phrase}, {insulin_phrase}, and {comorb_phrase}. "
        f"{anatomy} Conditions include: {conditions}."
    )

    # Compose the final prompt
    return f"""
{description}

Based on the provided patient information, does the patient has Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()



def mBRSET_ONLY_TEXT_PROMPT(row):
    # Age
    age_phrase = (
        f"aged {row['age']} years" 
        if not pd.isnull(row['age']) 
        else "with age not reported"
    )

    # Diabetes duration
    diabetes_phrase = (
        f"diagnosed with diabetes for {row['dm_time']} years" 
        if not pd.isnull(row['dm_time']) 
        else "with no reported diabetes duration"
    )

    # Educational level
    education = education_map.get(row['educational_level'], "with no educational level reported")

    # Build descriptions
    sex = convert_sex_mbrset(row['sex'])
    insulin = binary_to_text(row['insulin'], "using insulin", "not using insulin")
    oral = binary_to_text(row['oraltreatment_dm'], "on oral treatment for diabetes", "not on oral treatment for diabetes")
    hypertension = binary_to_text(row['systemic_hypertension'], "with systemic hypertension", "without systemic hypertension")
    alcohol = binary_to_text(row['alcohol_consumption'], "consumes alcohol", "does not consume alcohol")
    smoking = binary_to_text(row['smoking'], "smokes", "does not smoke")
    obesity = binary_to_text(row['obesity'], "with obesity", "without obesity")
    vascular = binary_to_text(row['vascular_disease'], "has vascular disease", "does not have vascular disease")
    infarction = binary_to_text(row['acute_myocardial_infarction'], "has a history of acute myocardial infarction", "no history of acute myocardial infarction")
    nephropathy = binary_to_text(row['nephropathy'], "with nephropathy", "without nephropathy")
    neuropathy = binary_to_text(row['neuropathy'], "with neuropathy", "without neuropathy")
    diabetic_foot = binary_to_text(row['diabetic_foot'], "has diabetic foot", "does not have diabetic foot")

    # Compose patient description
    description = (
        f"A {sex} patient {age_phrase}, {diabetes_phrase}, {insulin}, and {oral}. "
        f"The patient is {hypertension}, {alcohol}, {smoking}, {obesity}, and {vascular}. "
        f"Medical history includes: {infarction}, {nephropathy}, {neuropathy}, and {diabetic_foot}. "
        f"The patient is {education}."
    )

    # LLM prompt
    return f"""
{description}

Based on the provided patient information, does the patient have Diabetic Retinopathy (DR)?

Respond with **yes** if the patient has any level of diabetic retinopathy (ICDR score ≥ 1), or **no** if the score is 0. 
According to the International Clinical Diabetic Retinopathy (ICDR) classification, an eye is considered ICDR 0 when no retinal abnormalities related to diabetic retinopathy are present. ICDR ≥1 indicates the presence of any diabetic retinopathy, defined by the observation of one or more characteristic lesions such as microaneurysms, intraretinal hemorrhages, hard exudates,  venous beading, intraretinal microvascular abnormalities (IRMA), neovascularization, or vitreous/preretinal hemorrhage. Additionally, the presence of panretinal (panphotocoagulation) laser scars is considered evidence of treated proliferative diabetic retinopathy.

Respond only with "yes" or "no" (without additional commentary).
""".strip()


# New prompts for specific tasks requested by user

def BRSET_5_CLASS_DR_PROMPT(row):
    description = BRSET_TEXT_PROMPT(row).split("\n\nBased on")[0] # Reuse description logic
    return f"""
{description}

Based on the patient information and fundus image, what is the severity grade of Diabetic Retinopathy?

Respond with a single number:
0: No DR
1: Mild NPDR
2: Moderate NPDR
3: Severe NPDR
4: Proliferative DR

Answer only with the number (0-4).
""".strip()

def BRSET_3_CLASS_DR_PROMPT(row):
    description = BRSET_TEXT_PROMPT(row).split("\n\nBased on")[0]
    return f"""
{description}

Based on the patient information and fundus image, classify the Diabetic Retinopathy severity.

Respond with a single number:
0: No DR
1: Non-Proliferative DR (Mild, Moderate, Severe)
2: Proliferative DR

Answer only with the number (0-2).
""".strip()

def mBRSET_3_CLASS_DR_PROMPT(row):
    description = mBRSET_TEXT_PROMPT(row).split("\n\nBased on")[0]
    return f"""
{description}

Based on the patient information and fundus image, classify the Diabetic Retinopathy (DR) severity into one of the following 3 classes:

0: No DR (ICDR 0)
1: Non-Proliferative DR (ICDR 1-3: Mild, Moderate, Severe)
2: Proliferative DR (ICDR 4)

Respond only with the class number: **0**, **1**, or **2**.
""".strip()

def BRSET_GLAUCOMA_PROMPT(row):
    description = BRSET_TEXT_PROMPT(row).split("\n\nBased on")[0]
    return f"""
{description}

Based on the patient information and fundus image, does the patient have Glaucoma (or increased cup-disc ratio indicative of glaucoma)?

Respond with **yes** or **no**.
""".strip()

def BRSET_AMD_PROMPT(row):
    return f"""
Based on the image, does the patient has Age-related Macular Degeneration (AMD)?

Respond with **yes** if the patient has AMD, or **no** otherwise.

Respond only with "yes" or "no" (without additional commentary).
""".strip()


def BRSET_BINARY_DR_PROMPT(row):
    # This is effectively the same as BRSET_TEXT_PROMPT
    return BRSET_TEXT_PROMPT(row)

def mBRSET_BINARY_DR_PROMPT(row):
    # This is effectively the same as mBRSET_TEXT_PROMPT
    return mBRSET_TEXT_PROMPT(row)

def BRSET_REFERABLE_DR_PROMPT(row):
    description = BRSET_TEXT_PROMPT(row).split("\n\nBased on")[0]
    return f"""
{description}

Based on the provided patient information and the associated fundus image, does the patient has Referable Diabetic Retinopathy?

Respond with **yes** if the patient has referable diabetic retinopathy (Moderate NPDR or worse, ICDR score ≥ 2, or presence of Diabetic Macular Edema), or **no** otherwise.

Respond only with "yes" or "no" (without additional commentary).
""".strip()

def mBRSET_REFERABLE_DR_PROMPT(row):
    description = mBRSET_TEXT_PROMPT(row).split("\n\nBased on")[0]
    return f"""
{description}

Based on the provided patient information and the associated fundus image, does the patient has Referable Diabetic Retinopathy?

Respond with **yes** if the patient has moderate or worse diabetic retinopathy (ICDR score ≥ 2) or diabetic macular edema, or **no** otherwise.

Respond only with "yes" or "no" (without additional commentary).
""".strip()


######################## CoT Prompts #########################

def BRSET_BINARY_DR_COT_PROMPT(row):
    base_prompt = BRSET_BINARY_DR_PROMPT(row)
    # Remove the constraint "Respond only with..." to allow for reasoning
    prompt_lines = base_prompt.split('\n')
    # Filter out the constraint line if it matches closely
    filtered_lines = [line for line in prompt_lines if 'Respond only with "yes" or "no"' not in line]
    base_prompt = "\n".join(filtered_lines).strip()
    
    return f"""
{base_prompt}

Let's think step by step to determine the presence of Diabetic Retinopathy. Analyze the patient's clinical features and the fundus image details.
Finally, provide your answer as "Answer: yes" or "Answer: no".
""".strip()

def mBRSET_BINARY_DR_COT_PROMPT(row):
    base_prompt = mBRSET_BINARY_DR_PROMPT(row)
    prompt_lines = base_prompt.split('\n')
    filtered_lines = [line for line in prompt_lines if 'Respond only with "yes" or "no"' not in line]
    base_prompt = "\n".join(filtered_lines).strip()

    return f"""
{base_prompt}

Let's think step by step to determine the presence of Diabetic Retinopathy. Analyze the patient's clinical features and the fundus image details.
Finally, provide your answer as "Answer: yes" or "Answer: no".
""".strip()

def BRSET_REFERABLE_DR_COT_PROMPT(row):
    base_prompt = BRSET_REFERABLE_DR_PROMPT(row)
    prompt_lines = base_prompt.split('\n')
    filtered_lines = [line for line in prompt_lines if 'Respond only with "yes" or "no"' not in line]
    base_prompt = "\n".join(filtered_lines).strip()

    return f"""
{base_prompt}

Let's think step by step to determine if the Diabetic Retinopathy is referable. Consider the severity levels and presence of edema.
Finally, provide your answer as "Answer: yes" or "Answer: no".
""".strip()

def mBRSET_REFERABLE_DR_COT_PROMPT(row):
    base_prompt = mBRSET_REFERABLE_DR_PROMPT(row)
    prompt_lines = base_prompt.split('\n')
    filtered_lines = [line for line in prompt_lines if 'Respond only with "yes" or "no"' not in line]
    base_prompt = "\n".join(filtered_lines).strip()

    return f"""
{base_prompt}

Let's think step by step to determine if the Diabetic Retinopathy is referable. Consider the severity levels and presence of edema.
Finally, provide your answer as "Answer: yes" or "Answer: no".
""".strip()


######################## Role Prompts #########################

ROLE_HEADER = "You are an expert ophthalmologist specialized in diagnosing retinal diseases from fundus images and clinical metadata."

def BRSET_BINARY_DR_ROLE_PROMPT(row):
    return f"{ROLE_HEADER}\n\n{BRSET_BINARY_DR_PROMPT(row)}"

def mBRSET_BINARY_DR_ROLE_PROMPT(row):
    return f"{ROLE_HEADER}\n\n{mBRSET_BINARY_DR_PROMPT(row)}"

def BRSET_REFERABLE_DR_ROLE_PROMPT(row):
    return f"{ROLE_HEADER}\n\n{BRSET_REFERABLE_DR_PROMPT(row)}"

def mBRSET_REFERABLE_DR_ROLE_PROMPT(row):
    return f"{ROLE_HEADER}\n\n{mBRSET_REFERABLE_DR_PROMPT(row)}"


######################## Dispatcher #########################

def get_prompt_func(dataset_name, task, strategy="base"):
    """
    Returns the appropriate prompt function based on dataset, task, and strategy.
    """
    dataset_name = dataset_name.lower()
    task = task.lower()
    strategy = strategy.lower()

    # Base + Strategy Map construction
    # We can key by (dataset, task, strategy)
    
    mapping = {
        # Base
        ("brset", "binary_dr", "base"): BRSET_BINARY_DR_PROMPT,
        ("mbrset", "binary_dr", "base"): mBRSET_BINARY_DR_PROMPT,
        ("brset", "referable_dr", "base"): BRSET_REFERABLE_DR_PROMPT,
        ("mbrset", "referable_dr", "base"): mBRSET_REFERABLE_DR_PROMPT,
        
        # CoT
        ("brset", "binary_dr", "cot"): BRSET_BINARY_DR_COT_PROMPT,
        ("mbrset", "binary_dr", "cot"): mBRSET_BINARY_DR_COT_PROMPT,
        ("brset", "referable_dr", "cot"): BRSET_REFERABLE_DR_COT_PROMPT,
        ("mbrset", "referable_dr", "cot"): mBRSET_REFERABLE_DR_COT_PROMPT,
        
        # Role
        ("brset", "binary_dr", "role"): BRSET_BINARY_DR_ROLE_PROMPT,
        ("mbrset", "binary_dr", "role"): mBRSET_BINARY_DR_ROLE_PROMPT,
        ("brset", "referable_dr", "role"): BRSET_REFERABLE_DR_ROLE_PROMPT,
        ("mbrset", "referable_dr", "role"): mBRSET_REFERABLE_DR_ROLE_PROMPT,
    }

    # Also include existing tasks for base just in case, mapping them to 'base' logic if needed
    # But usually users might only ask strategies for the main DR tasks.
    
    key = (dataset_name, task, strategy)
    if key in mapping:
        return mapping[key]
        
    print(f"Warning: Prompt function not found for {key}. Trying fallback to base strategy.")
    key_base = (dataset_name, task, "base")
    if key_base in mapping:
        return mapping[key_base]

    # If it's another task (like glaucoma) that we haven't implemented strategies for yet, 
    # we might need to rely on the manual dispatcher in evaluate_retina.py or expand this.
    return None
