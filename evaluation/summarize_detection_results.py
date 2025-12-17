import csv

with open("results/Learned_versus_naive_phosphene/detection_summary.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(
            f"{row['Method']}: "
            f"{row['Detected']} / {row['Total Images']} "
            f"({float(row['Detection Rate'])*100:.1f}%)"
        )
'''
Output using this in the terminal instead of simply running the file: 
python -m evaluation.summarize_detection_results
'''