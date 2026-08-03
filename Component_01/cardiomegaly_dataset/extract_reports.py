import csv
import os

def main():
    csv.field_size_limit(10 * 1024 * 1024) # 10MB limit to handle large text fields

    base_dir = r"c:\Users\94775\Desktop\cardiomegaly_dataset"
    input_file = os.path.join(base_dir, "cardio_train.csv")
    out_dir = os.path.join(base_dir, "reports")
    pos_dir = os.path.join(out_dir, "positive")
    neg_dir = os.path.join(out_dir, "negative")

    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    pos_count = 0
    neg_count = 0
    seen_studies = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Determine column indices from header
        # Usually: 2=study_id, 5=Cardiomegaly, 18=findings_text, 19=report_text
        study_idx = header.index('study_id')
        cardio_idx = header.index('Cardiomegaly')
        findings_idx = header.index('findings_text')
        report_idx = header.index('report_text')

        for row in reader:
            if len(row) <= max(study_idx, cardio_idx, findings_idx, report_idx):
                continue
                
            study_id = row[study_idx]
            cardiomegaly = row[cardio_idx]
            findings = row[findings_idx]
            report = row[report_idx]

            # Skip if we've already processed this study to avoid duplicates
            if study_id in seen_studies:
                continue

            # Format the output as requested text format
            content = f"Findings:\n{findings.strip()}\n\n{'='*40}\n\nReport:\n{report.strip()}\n"

            if cardiomegaly == '1' and pos_count < 50:
                with open(os.path.join(pos_dir, f"{study_id}.txt"), 'w', encoding='utf-8') as out_f:
                    out_f.write(content)
                pos_count += 1
                seen_studies.add(study_id)
                
            elif cardiomegaly == '0' and neg_count < 50:
                with open(os.path.join(neg_dir, f"{study_id}.txt"), 'w', encoding='utf-8') as out_f:
                    out_f.write(content)
                neg_count += 1
                seen_studies.add(study_id)

            if pos_count >= 50 and neg_count >= 50:
                break

    print(f"Extraction complete! Saved {pos_count} positive and {neg_count} negative reports.")

if __name__ == '__main__':
    main()
