# Lighten ML Assessment

## Overview

You will be working with the publicly available [MIMIC IV demo dataset](https://www.physionet.org/content/mimic-iv-demo/2.2/hosp/#files-panel) and the associated clinical notes. The clinical notes are in discharge_notes_demo.csv, linked with the data tables by subject_id. The data tables are found in the link above; please download the dataset. The goal of this assessment is to build a clinical records retrieval and clinical reasoning system to determine any clinical variables of interest for each patient based on all available patient data.

Each clinical variable of interest comes with a variable name, variable type (Date, or Binary or Categorical), an abstraction guideline (linked via Google Docs), and the specific data tables it might reference as input. The variables could have a nested structure. In the Google Docs link, we gave an example abstraction guideline for Myocardial Infarction (a Y/N binary variable), and an example abstraction guideline for the Date of Myocardial Infarction Onset (a date variable). You should only extract the Date of Myocardial Infarction Onset if Myocardial Infarction is Y (i.e., if the patient has Myocardial Infarction). If Myocardial Infarction is N (i.e., if the patient doesn't have Myocardial Infarction), then the associated Date of Myocardial Infarction Onset should be NA for that patient.

We supplied the abstraction guidelines for Myocardial Infarction and its Onset Date as examples to help you think through the more general problem. In general, you could be given a variable tree structure (e.g., Variable A, with children Variable B and C, where Variable C has children D and E, and all of them have their respective abstraction guidelines). The children variables would have dependencies on the parent variables. In this case, the Date of Myocardial Infarction Onset is a child variable of Myocardial Infarction, because it doesn't make sense to extract the Onset Date of Myocardial Infarction if the patient doesn't have Myocardial Infarction. But you can also imagine that variables such as Symptoms at Myocardial Infarction Onset to be a child variable of Myocardial Infarction Onset Date, where we might want to extract all symptoms available at Myocardial Infarction onset. 

Please read through the example abstraction guideline carefully, as it captures several nuances, such as the rules to handle multiple found dates, or how to handle partial dates, or the logic used to determine whether a patient has Myocardial Infarction that references both the data tables (in this particular example, the lab values found in [hosp/labevents.csv](https://www.physionet.org/content/mimic-iv-demo/2.2/hosp/labevents.csv.gz) in the downloaded data) as well as the clinical notes. 

In this particular example, both Myocardial Infarction and the Onset Date of Myocardial Infarction would only require [hosp/labevents.csv](https://www.physionet.org/content/mimic-iv-demo/2.2/hosp/labevents.csv.gz) and [hosp/d_labitems.csv](https://www.physionet.org/content/mimic-iv-demo/2.2/hosp/d_labitems.csv.gz) from the data tables. [hosp/labevents.csv](https://www.physionet.org/content/mimic-iv-demo/2.2/hosp/labevents.csv.gz) only uses `itemid` to specify the lab, but you can find the `itemid` for Troponin T (a lab value referenced in the abstraction guidelines for Myocardial Infarction and the Onset Date of Myocardial Infarction) in [hosp/d_labitems.csv](https://www.physionet.org/content/mimic-iv-demo/2.2/hosp/d_labitems.csv.gz).

We are looking for a generalizable solution to reason and abstract any clinical variable of interest. Your code should take in patient data (both data tables and linked clinical notes), a nested tree variable structure, and the associated abstraction guideline, variable name, variable type and the specific data table names that are relevant for each variable in the tree as input, and output the relevant source evidences from the clinical notes as well as the correct clinical variable value for the patient for each of specified variables. 

## Instructions

#### Data Download and Data Parsing:
Please download the data tables using this link https://www.physionet.org/content/mimic-iv-demo/2.2/. Alternatively, you can run the following command in your terminal to download the dataset. `wget -r -N -c -np https://physionet.org/files/mimic-iv-demo/2.2/`

Please also parse discharge_notes_demo.csv, which includes clinical notes for the patients that are linked with the data tables via the column `subject_id`.

#### Retrieval and Reasoning
Build a generalizable system that takes in any nested tree of variables and patient data (including both the data tables, and unstructured clinical notes) as input, and output the correct value for each of the variables in the tree, by parsing through all relevant data tables and unstructured clinical notes and following the abstraction guidelines supplied for each of the variables. Your system should also output the specific evidences found in the raw data (whether it's from a data table, or from clinical notes). You can also expect that every variable comes with a variable name, type of variable (whether it's a Date, or binary, or categorical, etc.), which data tables are relevant to determine the value for this variable, and its associated abstraction guideline which could reference both the data tables and the clinical notes. Please note that in general, one patient's clinical notes can include thousands of separate notes, adding to up to ten thousand pages long. \

For this particular Myocardial Infarction example, here would be the example expected input and output:\
Example Input:\
Variable name: Myocardial Infarction \
Variable type: Binary \
Abstraction guideline: [insert linked abstraction guideline for myocardial infarction] \
Relevant data table names: hosp/labevents.csv, hosp/d_labitems.csv \
Children variables: [Myocardial Infarction Onset Date] 

Variable name: Myocardial Infarction Onset Date \
Variable type: Date\
Abstraction guideline: [insert linked abstraction guideline for myocardial infarction onset date] \
Relevant data table names: hosp/labevents.csv, hosp/d_labitems.csv \
Children variables:[] 

Please note that there can be many different data structures to represent the input; please suggest your own. 

Example Output: a nested JSON. To illustrate the nested structure, we're giving a more complicated example that assumes Myocardial Infarction Date has another child called Symptoms, where a patient could exhibit multiple symptoms at the onset of Myocardial Infarction. Note that "1001", "2001", "3001" are patient IDs (`subject_id`) This is for illustrate purpose only.
```javascript
{
    "1001": { 
        "Myocardial Infarction": [
            { 
                "value": "Y", 
                "Myocardial Infarction Date": [
                    {
                        "value": "2020-10-03",
                        "Symptoms": [
                            {"value": "fatigue"},
                            {"value": "chest pain"}
                        ]
                    }
                ]
            }
        ]
    },
    "2001": {
        "Myocardial Infarction": [
            {
                "value": "N",
                "Myocardial Infarction Date": [
                    {
                        "value": None,
                        "Symptoms": []
                    }
                ]
            }
        ]
    },
    "3001": {
        "Myocardial Infarction": [
            {
                "value": "Y",
                "Myocardial Infarction Date": [
                    {
                        "value": "2021-01-01",
                        "Symptoms": [
                            {"value": "fever"},
                            {"value": "nausea"}
                        ]
                    }
                ]
            }
        ]
    }
}
```
Alternatively, you could flatten this JSON into a CSV as the output. 



#### System Design
Your first task is to design a ML engineering system to handle this task. You will talk through your system design with a member of the Lighten team before you start any implementation. In general, you can expect between 500 to 10,000 pages of clinical notes per patient, consisting of hundreds of separate notes, and you can expect a typical abstraction guideline to be between 1-10 pages. These guidelines are written by our clinical team, and need to be followed precisely in order to achieve high quality results. For a single patient, we typically abstract 5 to 50 variables, many of them are in the nested tree structure as described above. 

#### Implementation
After the System Design conversation, you will use the remaining time for implementation. You can decide to implement one module of your system design, or you can decide to implement a MVP version of your entire system design. You will discuss your plan with the Lighten team member before you start implementation. Bonus if you can make at least part of your implementation to be production quality. 

You are encouraged to make LLM API calls, and will be given an API key to together.ai. Feel free to use your own API keys as well.

Email back your final code folder as a ZIP, and please include any documentations. Please feel free to ask any clarifications in the meantime. 














