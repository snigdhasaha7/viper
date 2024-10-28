import csv
import ast
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(filename='results_log.txt', level=logging.INFO,
                    format='%(asctime)s %(message)s')

def load_csv(filepath):
    """
    Load CSV data from the specified filepath.
    """
    data = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def is_correct(result, possible_answers):
    """
    Check if the result is in the possible_answers set.
    """
    # Convert string to a list
    possible_answers_list = ast.literal_eval(possible_answers)
    # Split the result into a list of items
    result_items = [item.strip().lower() for item in result.split('\n') if item.strip()]
    
    # Check if the result items match any possible answers
    return all(any(item in answer.lower() for answer in possible_answers_list) for item in result_items)

def evaluate_results(data, json_filepath='results.jsonl', incorrect_filepath='incorrect_results.txt'):
    """
    Evaluate the accuracy of each result in the CSV data.
    Log correct and incorrect results.
    """
    correct_count = 0
    incorrect_count = 0
    score = 0

    with open(json_filepath, 'w', encoding='utf-8') as jsonl_file, \
         open(incorrect_filepath, 'w', encoding='utf-8') as incorrect_file:

        for row in data:
            result = row['result']
            possible_answers = row['answer']
            episode_id = row['id']
            query = row['query']
            code = row.get('code', '')
            accuracy = float(row['accuracy'])
            img_path = row.get('img_path', '')

            if not possible_answers:
                continue

            score += accuracy

            if accuracy > 0.:
                logging.info(f'Episode {episode_id} - Correct: {query}')
                correct_count += 1
            else:
                logging.info(f'Episode {episode_id} - Incorrect: {query}; Predicted: {result}, Possible Answers: {possible_answers}')
                incorrect_count += 1

                # Write to incorrect_results.txt
                incorrect_file.write(f"Episode ID: {episode_id}\n")
                incorrect_file.write(f"Question: {query}\n")
                incorrect_file.write(f"Code: {code}\n")
                incorrect_file.write(f"Incorrect Result: {result}\n")
                incorrect_file.write(f"Possible Answers: {possible_answers}\n")
                incorrect_file.write("\n-----------------------------\n\n")

            # Collect data to output
            sample_data = {
                'accuracy': accuracy,
                'results': result,
                'answers': possible_answers,
                'episode_id': episode_id,
                'query': query,
                'code': code,
                'img_path': img_path
            }

            # Write JSON per sample
            json_line = json.dumps(sample_data, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')

    return correct_count, incorrect_count, score

def main():
    # Load CSV data
    filepath = '/home/gsarch/repo/viper/results/okvqav2/test/results_16_top1.csv'  # Replace with your CSV file path
    filepath = '/home/gsarch/repo/viper/results/okvqav2/test/results_17_top5.csv'  # Replace with your CSV file path
    # filepath = "/home/gsarch/repo/viper/results/okvqav2/test/results_2.csv"
    data = load_csv(filepath)

    json_filepath = f'results_21_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jsonl'
    incorrect_filepath = f'incorrect_results_21_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'

    # Evaluate the results
    correct_count, incorrect_count, score = evaluate_results(data, json_filepath, incorrect_filepath)

    # Log overall statistics
    total_items = correct_count + incorrect_count
    accuracy = (correct_count / (correct_count + incorrect_count)) * 100 if (correct_count + incorrect_count) > 0 else 0
    score = score / total_items
    logging.info(f'Total Episodes: {total_items}')
    logging.info(f'Correct Episodes: {correct_count}')
    logging.info(f'Incorrect Episodes: {incorrect_count}')
    logging.info(f'Accuracy: {accuracy:.2f}%')

    # Print stats to the console
    print(f"Total Episodes: {total_items}")
    print(f"Correct Episodes: {correct_count}")
    print(f"Incorrect Episodes: {incorrect_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Score: {score:.2f}")

if __name__ == '__main__':
    main()
