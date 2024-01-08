import json

class RareDataset():
    def __init__(self, dataset_name, dataset_path, dataset_type) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        if dataset_type == "EHR" or dataset_type == "PHENOTYPE":
            self.patient = self.load_ehr_phenotype_data()
        elif dataset_type == "MDT":
            self.patient = self.load_mdt_data()

    def load_ehr_phenotype_data(self):
        phenotype_mapping = json.load(open("mapping/phenotype_mapping.json", "r", encoding="utf-8-sig"))
        disease_mapping = json.load(open("mapping/disease_mapping.json", "r", encoding="utf-8-sig"))

        patient = []
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        for p in data:
            phenotype_list = p[0]
            disease_list = p[1]
            if self.dataset_type == "PHENOTYPE":
                phenotype_list = [phenotype_mapping[phenotype] for phenotype in phenotype_list if phenotype in phenotype_mapping]
                disease_list = [disease_mapping[disease] for disease in disease_list if disease in disease_mapping]
            phenotype = ",".join(phenotype_list)
            disease = ",".join(disease_list)
            patient.append((phenotype, disease))
            
        return patient
    
    def load_hpo_code_data(self):
        disease_mapping = json.load(open("mapping/disease_mapping.json", "r", encoding="utf-8-sig"))
        patient = []
        
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        for p in data:
            phenotype_list = p[0]
            disease_list = p[1]
            disease_list = [disease_mapping[disease] for disease in disease_list if disease in disease_mapping]
            disease = ",".join(disease_list)
            patient.append((phenotype_list, disease))
        return patient

    def load_mdt_data(self):
        patient = []
        with open(self.dataset_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        for k, v in data.items():
            ehr_info = v['二、病例介绍']
            ehr_info = "".join(ehr_info)
            golden_diagnosis = v['golden_diagnosis']
            
            patient.append((ehr_info, golden_diagnosis))
        return patient