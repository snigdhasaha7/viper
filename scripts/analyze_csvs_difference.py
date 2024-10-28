import csv
import logging
import json
import ast
from datetime import datetime

# Configure logging
logging.basicConfig(filename='results_log.txt', level=logging.INFO,
                    format='%(asctime)s %(message)s')

def load_csv(filepath):
    """
    Load CSV data from the specified filepath.
    Returns a dictionary mapping episode_id to row data.
    """
    data = {}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            episode_id = row['id']
            data[episode_id] = row
        return data

def parse_accuracy(accuracy_str):
    """
    Parse the accuracy string into a list of floats and return the maximum value.
    """
    try:
        accuracy_list = ast.literal_eval(accuracy_str)
        return max([float(a) for a in accuracy_list])
    except (ValueError, SyntaxError):
        return 0.0  # Default to 0 if parsing fails

def parse_code(code_str):
    """
    Parse the code string into a list.
    """
    
    code_list = ast.literal_eval(code_str)
    return code_list

def compare_and_log_differences(data1, data2, json_filepath='results.jsonl', differences_filepath='differences_log.txt'):
    """
    Compare the accuracies from two datasets and log differences for episodes where
    accuracy in data1 is less than accuracy in data2.
    """
    with open(json_filepath, 'w', encoding='utf-8') as jsonl_file, \
         open(differences_filepath, 'w', encoding='utf-8') as differences_file:

        for episode_id in data1:
            if episode_id in data2:
                row1 = data1[episode_id]
                row2 = data2[episode_id]

                # Parse accuracies
                accuracy1 = parse_accuracy(row1['accuracy'])
                accuracy2 = parse_accuracy(row2['accuracy'])

                if accuracy1 < accuracy2:
                    # Log the differences
                    logging.info(f'Episode {episode_id} - Accuracy in csv1 ({accuracy1}) < Accuracy in csv2 ({accuracy2})')

                    # Parse codes
                    code1_list = parse_code(row1.get('code', ''))
                    code2_list = parse_code(row2.get('code', ''))
                    

                    # Write differences to the text file
                    differences_file.write(f"Episode ID: {episode_id}\n")
                    differences_file.write(f"Accuracy csv1: {accuracy1}\n")
                    differences_file.write(f"Accuracy csv2: {accuracy2}\n")
                    differences_file.write(f"Question: {row1['query']}\n")
                    differences_file.write(f"Result csv1: {row1['result']}\n")
                    differences_file.write(f"Result csv2: {row2['result']}\n")
                    differences_file.write(f"Possible Answers: {row1['answer']}\n")

                    # Write code outputs from both CSVs
                    differences_file.write("\nCode Outputs from csv1:\n")
                    for idx, code in enumerate(code1_list, 1):
                        differences_file.write(f"Code {idx}:\n{code}\n\n")

                    differences_file.write("\nCode Outputs from csv2:\n")
                    for idx, code in enumerate(code2_list, 1):
                        differences_file.write(f"Code {idx}:\n{code}\n\n")

                    differences_file.write("\n-----------------------------\n\n")

                    # Collect data to output
                    sample_data = {
                        'episode_id': episode_id,
                        'accuracy_csv1': accuracy1,
                        'accuracy_csv2': accuracy2,
                        'query': row1['query'],
                        'result_csv1': row1['result'],
                        'result_csv2': row2['result'],
                        'possible_answers': row1['answer'],
                        'code_csv1': code1_list,
                        'code_csv2': code2_list,
                    }

                    # Write JSON per sample
                    json_line = json.dumps(sample_data, ensure_ascii=False)
                    jsonl_file.write(json_line + '\n')
            else:
                logging.warning(f'Episode {episode_id} is in csv1 but not in csv2')

def main():
    # Load CSV data
    filepath1 = '/home/gsarch/repo/viper/results/okvqav2/test/results_25_500_top1.csv'  # Replace with your first CSV file path
    filepath2 = '/home/gsarch/repo/viper/results/okvqav2/test/results_26_500_top5.csv'  # Replace with your second CSV file path

    data1 = load_csv(filepath1)
    data2 = load_csv(filepath2)

    json_filepath = f'results_difference_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jsonl'
    differences_filepath = f'differences_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'

    # Compare and log differences
    compare_and_log_differences(data1, data2, json_filepath, differences_filepath)

    print("Comparison complete. Differences logged.")

if __name__ == '__main__':
    main()
