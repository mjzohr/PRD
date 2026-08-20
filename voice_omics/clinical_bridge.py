"""Clinical-text prompt builder. Turns a metadata row into a natural-language
sentence encoded by the frozen clinical LM. Diagnosis words are used only for the
text anchor; the supervised label is always carried separately.

Sharp framing: GOLD 0 -> "Healthy", GOLD >=3 -> "Severe COPD", mirroring the
sharp-class teacher signal.
"""

import pandas as pd


class ClinicalTextBridge:
    def _get(self, row, keys, default="Unknown"):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return default

    def _demographics(self, row):
        age = self._get(row, ["age_visit", "age_baseline", "age", "Age_P2", "Age_P1", "Age"], default=None)
        age_str = f"{int(float(age))}-year-old" if age not in (None, "Unknown") else "Adult"
        sex_val = str(self._get(row, ["sex", "gender", "Gender", "Sex"], default="")).lower()
        if "f" in sex_val or sex_val == "2":
            sex_str = "female"
        elif "m" in sex_val or sex_val == "1":
            sex_str = "male"
        else:
            sex_str = "patient"
        return age_str, sex_str

    def _gold_phrase(self, row):
        gold = self._get(row, ["finalGold", "finalgold_visit", "finalGold_P1", "final_Gold_Stage"], default=None)
        try:
            gold = int(float(gold))
        except (TypeError, ValueError):
            return None
        if gold == 0:
            return "Healthy condition"
        if gold >= 3:
            return "Severe Chronic Obstructive Pulmonary Disease"
        return "Chronic Obstructive Pulmonary Disease"

    def generate_text(self, row, source_type="icbhi"):
        age_str, sex_str = self._demographics(row)

        if source_type == "icbhi":
            d = row.get("disease", "Unknown")
            disease_str = {
                "COPD": "Chronic Obstructive Pulmonary Disease",
                "Healthy": "Healthy condition",
            }.get(d, str(d))
            c, w = row.get("crackles", 0), row.get("wheezes", 0)
            if c == 1 and w == 1:
                sound = " Auscultation reveals crackles and wheezes."
            elif c == 1:
                sound = " Auscultation reveals crackles."
            elif w == 1:
                sound = " Auscultation reveals wheezes."
            else:
                sound = " Auscultation reveals normal breathing."
            return f"A {age_str} {sex_str} diagnosed with {disease_str}.{sound}"

        if source_type in ("omics", "ct_clinical"):
            disease_str = self._gold_phrase(row) or "respiratory condition"
            modality = "Clinical proteomics profile." if source_type == "omics" else "Chest CT and clinical profile."
            return f"A {age_str} {sex_str} presenting with {disease_str}. {modality}"

        return "Patient data."
