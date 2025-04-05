"""
Implementation of OpenAI's simpleQA to benchmark language models.
"""

import openai
import pandas as pd
from dotenv import load_dotenv
from google import genai
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def simpleQA(model_name, grader_model_name, test_set):
    """
    model_name: language model using openai api or gemini api
    grader_model_name: language model using openai api or gemini api
    test_set: list of dicts with keys 'metadata', 'problem', 'answer'

    Uses a grader model to grade the passed in model's performance.
    To grade questions, we use a prompted ChatGPT classifier that sees both the predicted answer from the model and the ground-truth answer, and then grades the predicted answer as either "correct", "incorrect", or "not attempted". 
    """

    for question in tqdm(test_set, desc=f"Evaluating {model_name}"):
        prompt = f"""
        Answer the following question, only return the answer followed by your confidence on a new line from 0 to 100.

        Examples:
        Question: What is the capital of France?
        Paris
        Confidence: 95

        Question: What is the largest planet in our solar system?
        Jupiter
        Confidence: 100

        Question: Who was the 23rd President of Madagascar?
        Didier Ratsiraka
        Confidence: 45

        Question: What is the exact population of Tokyo in 2024?
        Approximately 37.4 million
        Confidence: 60

        Question: What was the first movie ever made?
        Roundhay Garden Scene
        Confidence: 75

        Question: How many species of beetles exist?
        Around 400,000
        Confidence: 30

        ---

        Question: {question['problem']}
        """
        if model_name == 'gemini-2.0-flash':
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt]
            )
            raw_response = response.text
        else:
            response = openai.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_response = response.choices[0].message.content
        try:
            predicted_answer = raw_response.split("Confidence: ")[0].strip()
            confidence = raw_response.split("Confidence: ")[1].strip()
        except:
            predicted_answer = raw_response
            confidence = -1
        
        # Grade the predicted answer
        grader_prompt = f"""
        You are a grading model for simple QA. For each question and prediction, return one of: "Correct", "Incorrect", or "Not attempted".

        Examples:

        Question: Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men's FIFA World Cup?
        Reference answer: Wout Weghorst
        Predicted answer: Wout Weghorst
        Grade: Correct

        ---

        Question: Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men's FIFA World Cup?
        Reference answer: Wout Weghorst
        Predicted answer: Wout Weghorst scored at 83' and 90+11' in that game
        Grade: Correct

        ---

        Question: Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men's FIFA World Cup?
        Reference answer: Wout Weghorst
        Predicted answer: Virgil van Dijk
        Grade: Incorrect

        ---

        Question: Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men's FIFA World Cup?
        Reference answer: Wout Weghorst
        Predicted answer: Virgil van Dijk and Wout Weghorst
        Grade: Incorrect

        ---

        Question: Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men's FIFA World Cup?
        Reference answer: Wout Weghorst
        Predicted answer: Wout Weghorst and I think van Dijk scored, but I am not totally sure
        Grade: Incorrect

        ---

        Question: Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men's FIFA World Cup?
        Reference answer: Wout Weghorst
        Predicted answer: I don't know the answer to that question
        Grade: Not attempted

        ---

        Question: Which Dutch player scored an open-play goal in the 2022 Netherlands vs Argentina game in the men's FIFA World Cup?
        Reference answer: Wout Weghorst
        Predicted answer: To find which Dutch player scored in that game, please browse the internet yourself
        Grade: Not attempted

        ---

        Question: {question['problem']}
        Reference answer: {question['answer']}
        Predicted answer: {predicted_answer}
        Grade: 
        """
        
        if grader_model_name == 'gemini-2.0-flash':
            grader_response = client.models.generate_content(
                model=grader_model_name,
                contents=[grader_prompt]
            )
            grade = grader_response.text
        else:
            grader_response = openai.chat.completions.create(
                model=grader_model_name,
                messages=[
                    {"role": "system", "content": "You are a grading model for simple QA."},
                    {"role": "user", "content": grader_prompt}
                ]
            )
            grade = grader_response.choices[0].message.content
        if "Grade: " in grade:
            grade = grade.split("Grade: ")[1].strip()

        question['predicted_answer'] = predicted_answer
        question['confidence'] = confidence
        question['grade'] = grade

    return test_set

"""
Builds a dataframe of the results of the simpleQA test set for each model
"""
def evaluate_models(models, grader_model_name, test_set):
    results = []
    for model in models:
        model_results = simpleQA(model, grader_model_name, test_set)
        model_results = pd.DataFrame(model_results)
        model_results["model"] = model
        results.append(model_results)
    results = pd.concat(results)
    return results



"""
Plots the calibration curve for all models
"""
def plot_calibration_curve_all(results):
    # model calibration plot
    # y axis is accuracy, x axis is confidence bins

    # Set the style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Read data and preprocess
    results = results[results['confidence'] != -1]
    results['confidence'] = results['confidence'].astype(float) / 100
    results['correct'] = results['grade'] == 'Correct'

    # Define bins
    bins = np.linspace(0, 1, 11)

    # Create figure with seaborn styling
    plt.figure(figsize=(10, 6))

    # Plot line for each model
    for model in results['model'].unique():
        model_data = results[results['model'] == model]
        bin_indices = np.digitize(model_data['confidence'], bins, right=True)
        
        # Compute average confidence and accuracy per bin
        bin_confidences = []
        bin_accuracies = []
        
        for i in range(1, len(bins)):
            bin_data = model_data[bin_indices == i]
            if len(bin_data) == 0:
                continue
            bin_confidences.append(bin_data['confidence'].mean())
            bin_accuracies.append(bin_data['correct'].mean())
        
        sns.lineplot(x=bin_confidences, y=bin_accuracies, marker='o', label=model, linewidth=2, markersize=8)

    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)

    # Customize the plot
    plt.xlabel('Confidence', fontsize=12, labelpad=10)
    plt.ylabel('Accuracy', fontsize=12, labelpad=10)
    plt.title('Model Calibration Curves', fontsize=14, pad=20)

    # Customize legend
    plt.legend(title='Model', title_fontsize=12, fontsize=10, 
            bbox_to_anchor=(1.05, 1), loc='upper left',
            frameon=True, fancybox=True, shadow=True)

    # Set axis limits
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # Add minor gridlines
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)

    plt.tight_layout()
    plt.savefig('calibration_curve.png')


"""
Plots the bar plot of the model performance
"""
def plot_bar_plot(results):
    # Calculate percentages for each grade type per model
    results_pct = results.groupby('model')['grade'].value_counts(normalize=True).unstack() * 100

    # Set the style
    sns.set_style("whitegrid")

    # Create stacked bar chart
    ax = results_pct.plot(kind='bar', stacked=True, figsize=(10,6))

    # Use seaborn color palette
    colors = sns.color_palette("husl", n_colors=len(results_pct.columns))

    # Customize using seaborn
    plt.title('Model Performance on SimpleQA Test Set', pad=20)
    plt.xlabel('Model', labelpad=10)
    plt.ylabel('Percentage of Answers', labelpad=10)

    # Create legend with matching colors
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(results_pct.columns))]
    plt.legend(handles, results_pct.columns, title='Grade', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Color the bars and add labels
    for i, container in enumerate(ax.containers):
        for patch in container:
            patch.set_facecolor(colors[i])
        ax.bar_label(container, fmt='%.1f%%')

    plt.tight_layout()
    plt.savefig('bar_plot.png')

"""
Script to run the simpleQA test set
Allows the user to input names of models to test from the openai api and gemeni api
outputs a csv with the results and a bar plot of the model performance.

Usage:
    python test.py <output_file>
"""
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test.py <output_file>")
        sys.exit(1)
        
    output_file = sys.argv[1]
    models = ['gpt-4o-mini', 'gpt-4o', 'o3-mini', 'gemini-2.0-flash']
    grader_model_name = 'gpt-4o-mini'
    test_set = pd.read_csv('simple_qa_test_set.csv')
    test_set = test_set.to_dict(orient='records')
    test_set = test_set[:5]
    results = evaluate_models(models, grader_model_name, test_set)
    results.to_csv(output_file, index=False)
    plot_calibration_curve_all(results)
    plot_bar_plot(results)
