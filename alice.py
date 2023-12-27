import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

# Set background color and text color
st.markdown(
    """
    <style>
        .highlight {
            background-color: #DAA520;  /* Yellow color as an example */
            padding: 10px;
            border-radius: 5px;
        }
        .highlight h1 {
            color: #000000;  /* Black color for text */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add title with the highlighted background
st.markdown("<div class='highlight'><h1>Performance Metrics for Yandex  Alice</h1></div>", unsafe_allow_html=True)


st.subheader("Introduction")
st.write("In this project, I have selected key metrics such as accuracy, precision, recall, F1 score, confusion matrix, contextual relevance and human-like interaction to evaluate the performance of Alice, our chatbot. These chosen metrics provide valuable insights into the quality and reliability of Alice's interactions, ensuring a robust evaluation of its performance in real-world conversational scenarios. We also recognize the importance of human evaluation, understanding that factors like readability, coherence, and overall user satisfaction contribute significantly to assessing the success of a chatbot system.")

st.write("This dataset was created by me and Alice. I tried as much as possible to be objective in judgment.")

# Read csv and display df
csv_file_path = 'query-response.csv'
df = pd.read_csv(csv_file_path)
st.subheader("Original DataFrame")
st.write(df)

st.write("Here are 31 questions in Russian and 31 exact questions in Kazakh to observe how Alice responds to the same question in different languages. Our aim is not only to assess Alice's capabilities but also to examine how the Kazakh version differs from the original Russian version. The selected questions span 10 categories, representing queries that users might ask in their daily lives.")

# Divide
index_of_query = df[df['User Query'] == 'Kazakh language'].index

# Creating df with only Kazakh queries
qaz_queries = df[df.index >= index_of_query.min()]

qaz_queries.reset_index(drop=True, inplace=True)

# Creating df with only Russian queries
rus_queries = df[df.index < index_of_query.min()]

# Display the filtered DataFrame
st.subheader("Filtered DataFrame with only Russian language queries")
st.write(rus_queries)

# Display the filtered DataFrame
st.subheader("Filtered DataFrame with only Kazakh language queries")
st.write(qaz_queries)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# ACCURACY 
st.title("Accuracy")
st.write("Accuracy is a metric used to measure the effectiveness of Alice's responses. The formula for accuracy is calculated as the number of correct responses divided by the total number of responses.")
st.latex(r"\text{Accuracy} = \frac{\text{Correct Responses}}{\text{Total Responses}}")
st.write("For this project, <Yes> and <Satisfactory> responses in the <Is it a Perfect Response?> column are considered correct, while <Not Satisfactory> responses are treated as incorrect.")
st.write("The accuracy metric provides an overall percentage reflecting how often Alice's responses align with the desired criteria, offering a quantitative measure of the system's performance.")

# Accuracy calculation for Total 
correct_total = df["Is it perfect response?"].isin(["Yes", "Satisfactory"]).sum()
incorrect_total = df["Is it perfect response?"].isin(["Not satisfactory"]).sum()

accuracy_total = correct_total / (correct_total + incorrect_total)

# Russian DataFrame
correct_russian = rus_queries["Is it perfect response?"].isin(["Yes", "Satisfactory"]).sum()
incorrect_russian = rus_queries["Is it perfect response?"].isin(["Not satisfactory"]).sum()

accuracy_russian = correct_russian / (correct_russian + incorrect_russian)

# Kazakh DataFrame
correct_kazakh = qaz_queries["Is it perfect response?"].isin(["Yes", "Satisfactory"]).sum()
incorrect_kazakh = qaz_queries["Is it perfect response?"].isin(["Not satisfactory"]).sum()

accuracy_kazakh = correct_kazakh / (correct_kazakh + incorrect_kazakh)

# Display accuracy
st.subheader("Accuracy Progress Bars")

# Display the accuracy for total using a progress bar
st.write(f"Accuracy Total: {accuracy_total:.2%}")
st.progress(accuracy_total)

# Display the accuracy for Russian using a progress bar
st.write(f"Accuracy Russian language: {accuracy_russian:.2%}")
st.progress(accuracy_russian)

# Display the accuracy for Kazakh using a progress bar
st.write(f"Accuracy Kazakh language: {accuracy_kazakh:.2%}")
st.progress(accuracy_kazakh)

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Selectbox
accuracy_choice = st.selectbox("Choose Accuracy Data Frame:", ["Kazakh", "Russian", "Total"])

if accuracy_choice == "Kazakh":
    accuracy_df = qaz_queries
elif accuracy_choice == "Russian":
    accuracy_df = rus_queries
    pass
else:
    accuracy_df = df  # Total DataFrame

# Calculate accuracy
correct_rows = accuracy_df["Is it perfect response?"].isin(["Yes", "Satisfactory"]).sum()
incorrect_rows = accuracy_df["Is it perfect response?"].isin(["Not satisfactory"]).sum()

# Calculate accuracy
total_rows = correct_rows + incorrect_rows
accuracy = correct_rows / total_rows if total_rows > 0 else 0


# Display accuracy
st.subheader(f"Accuracy for {accuracy_choice} Data Frame:")
st.write(f"Correct Responses: {correct_rows}")
st.write(f"Total Responses: {total_rows}")
st.write(f"Accuracy: {accuracy:.2%}")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# PRECISION
st.title("Precision")
st.write("Precision is a metric evaluating the accuracy of positive predictions made by a model. In the context of this project, precision gauges the accuracy of Alice's responses when predicting a positive outcome, such as delivering a satisfactory or correct response. A higher precision signifies that Alice's positive predictions are more trustworthy and less likely to include false positives.")
st.latex(r"\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Calculate Precision for each data frame
TP_total = (df['True Positive'] == True ).sum()
FP_total = (df['False Positive'] == True ).sum()

precision_total = TP_total / (TP_total + FP_total)

TP_rus = (rus_queries['True Positive'] == True ).sum()
FP_rus = (rus_queries['False Positive'] == True ).sum()

precision_russian = TP_rus / (TP_rus + FP_rus)

TP_qaz = (qaz_queries['True Positive'] == True ).sum()
FP_qaz = (qaz_queries['False Positive'] == True ).sum()

precision_kazakh = TP_qaz / (TP_qaz + FP_qaz)

# Display Precision
st.subheader("Precision for Different Data Frames:")

# Specify the y-axis range to focus on the relevant percentage range
y_axis_range = [min(precision_kazakh, precision_russian, precision_total) * 100 - 5, 100]

# Set the bar width
bar_width = 0.5

# Create a figure
precision_fig = go.Figure()

# Add bar traces
for i, label in enumerate(["Total", "Kazakh", "Russian"]):
    precision_fig.add_trace(
        go.Bar(
            x=[label],
            y=[[precision_total * 100, precision_kazakh * 100, precision_russian * 100][i]],
            name=label,
            marker_color=['#FFD700', '#00A1DE', '#FF6666'][i],
            width=bar_width,
        )
    )

# Update layout
precision_fig.update_layout(
    title="Precision Distribution",
    yaxis=dict(title="Precision (%)", range=y_axis_range),
)

# Show the plot
st.plotly_chart(precision_fig)


# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# RECALL
st.title("Recall")
st.write("Recall, also known as sensitivity or true positive rate, assesses the model's ability to correctly identify all relevant instances, specifically the true positives. In the context of this project, recall evaluates how well Alice can capture and retrieve all correct responses to user queries. A higher recall indicates that Alice is effective at retrieving relevant responses, even if there are some false negatives.")
st.latex(r"\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Calculate recall for each data frame
TP_total = (df['True Positive'] == True ).sum()
FN_total = (df['False Negative'] == True ).sum()

recall_total = TP_total / (TP_total + FN_total)

TP_rus = (rus_queries['True Positive'] == True ).sum()
FN_rus = (rus_queries['False Negative'] == True ).sum()

recall_russian = TP_rus / (TP_rus + FN_rus)

TP_qaz = (qaz_queries['True Positive'] == True ).sum()
FN_qaz = (qaz_queries['False Negative'] == True ).sum()

recall_kazakh = TP_qaz / (TP_qaz + FN_qaz)

# Display recall
st.subheader("Recall for Different Data Frames:")

# Specify the y-axis range to focus on the relevant percentage range
y_axis_range = [min(recall_kazakh, recall_russian, recall_total) * 100 - 15, 100]

# Set the bar width
bar_width = 0.5

precision_fig = go.Figure()

# Add bar traces
for i, label in enumerate(["Total", "Kazakh", "Russian"]):
    precision_fig.add_trace(
        go.Bar(
            x=[label],
            y=[[recall_total * 100, recall_kazakh * 100, recall_russian * 100][i]],
            name=label,
            marker_color=['#FFD700', '#00A1DE', '#FF6666'][i],
            width=bar_width,
        )
    )

# Update layout
precision_fig.update_layout(
    title="Precision Distribution",
    yaxis=dict(title="Precision (%)", range=y_axis_range),
)

# Show the plot
st.plotly_chart(precision_fig)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# F1 SCORE
st.title("F1 score")
st.write("The F1 score, also known as the F1 measure or F1 value, is a metric that combines precision and recall into a single value. It provides a balance between the two, making it useful in scenarios where there is an uneven class distribution. The formula for F1 score is given by:")

# Formula
f1_formula = r"F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}"
# Display the formula using st.latex()
st.latex(f1_formula)

st.write("The F1 score ranges between 0 and 1, with higher values indicating better balance between precision and recall. It is particularly valuable in binary classification tasks where both false positives and false negatives need to be considered.")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("F1 Score Progress Bars")

# Total
f1_total = 2 * precision_total * recall_total / (precision_total + recall_total)

st.write(f"F1 Score Total: {f1_total:.2%}")
# Display the F1 score using a progress bar
st.progress(f1_total)

# Russian 
f1_russian = 2 * precision_russian * recall_russian / (precision_russian + recall_russian)

st.write(f"F1 Score Russian language: {f1_russian:.2%}")
# Display the F1 score using a progress bar
st.progress(f1_russian)

# Kazakh 
f1_kazakh = 2 * precision_kazakh * recall_kazakh / (precision_kazakh + recall_kazakh)

st.write(f"F1 Score Kazakh: {f1_kazakh:.2%}")
# Display the F1 score using a progress bar
st.progress(f1_kazakh)

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

st.title("Confusion Matrix")
st.write("A confusion matrix is a table that is often used to evaluate the performance of a classification model. It provides a detailed breakdown of the model's predictions and their actual outcomes. In the context of this project, where Alice's responses are being evaluated, the confusion matrix helps analyze the performance of the system in terms of true positives (correctly identified positive responses), true negatives (correctly identified negative responses), false positives (incorrectly identified positive responses), and false negatives (incorrectly identified negative responses).")
st.write("The confusion matrix provides a comprehensive view of how well Alice's responses align with the actual correctness, aiding in a deeper understanding of the system's strengths and weaknesses. The absence of true negatives in this context is due to the nature of the evaluation, where the goal is to assess the correctness of responses rather than a binary classification scenario.")
st.write("Total responses")

# Assuming you have the following values
TP = (df['True Positive'] == True ).sum()
FP = (df['False Positive'] == True ).sum()
FN = (df['False Negative'] == True ).sum()

# Create a DataFrame for the confusion matrix
confusion_matrix_data = pd.DataFrame({
    'Actual Negative': [0, FP],
    'Actual Positive': [FN, TP]
}, index=['Predicted Negative', 'Predicted Positive'])

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix for Total Responses')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Display the plot in Streamlit
st.pyplot(plt)

# Russian language
st.write("Russian language responses")

# Assuming you have the following values
TP = (rus_queries['True Positive'] == True ).sum()
FP = (rus_queries['False Positive'] == True ).sum()
FN = (rus_queries['False Negative'] == True ).sum()

# Create a DataFrame for the confusion matrix
confusion_matrix_data = pd.DataFrame({
    'Actual Negative': [0, FP],
    'Actual Positive': [FN, TP]
}, index=['Predicted Negative', 'Predicted Positive'])

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix for Russian language responses')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Display the plot in Streamlit
st.pyplot(plt)

# Kazakh language
st.write("Kazakh language responses")

# Assuming you have the following values
TP = (qaz_queries['True Positive'] == True ).sum()
FP = (qaz_queries['False Positive'] == True ).sum()
FN = (qaz_queries['False Negative'] == True ).sum()

# Create a DataFrame for the confusion matrix
confusion_matrix_data = pd.DataFrame({
    'Actual Negative': [0, FP],
    'Actual Positive': [FN, TP]
}, index=['Predicted Negative', 'Predicted Positive'])

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix for Kazakh language responses')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Display the plot in Streamlit
st.pyplot(plt)

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# CONTEXTUAL RELEVANCE 
st.title("Contextual Relevance")
st.write("Contextual Relevance measures how well Alice's responses align with the context or meaning of the user's queries. In the context of this project, contextual relevance is assessed based on the qualitative judgment of human evaluators, considering whether Alice's responses are appropriate and contextually aligned with the given queries.")
st.latex(r"\text{Contextual Relevance} = \frac{\text{Number of Contextually Relevant Responses}}{\text{Total Number of Responses}} \times 100")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Total
# Trimming spaces in answers
df["Contextually relevant"] = df["Contextually relevant"].apply(lambda x: x.strip() if isinstance(x, str) else x)

rel_total = (df['Contextually relevant'] == 'Yes' ).sum()
not_rel_total = (df['Contextually relevant'] == 'No' ).sum()

context_total = rel_total / (rel_total + not_rel_total)

# Russian
rus_queries["Contextually relevant"] = rus_queries["Contextually relevant"].apply(lambda x: x.strip() if isinstance(x, str) else x)

rel_rus = (rus_queries['Contextually relevant'] == 'Yes' ).sum()
not_rel_rus = (rus_queries['Contextually relevant'] == 'No' ).sum()

context_rus = rel_rus / (rel_rus + not_rel_rus)

# Kazakh
qaz_queries["Contextually relevant"] = qaz_queries["Contextually relevant"].apply(lambda x: x.strip() if isinstance(x, str) else x)

rel_qaz = (qaz_queries['Contextually relevant'] == 'Yes' ).sum()
not_rel_qaz = (qaz_queries['Contextually relevant'] == 'No' ).sum()

context_qaz = rel_qaz / (rel_qaz + not_rel_qaz)

# Display contextual relevance
st.subheader("Contextual Relevance Progress Bars")

# Display Contextual Relevance for total using a progress bar
st.write(f"Contextual Relevance Total: {context_total:.2%}")
st.progress(context_total)

# Display Contextual Relevance for Russian using a progress bar
st.write(f"Contextual Relevance Russian language: {context_rus:.2%}")
st.progress(context_rus)

# Display Contextual Relevance for Kazakh using a progress bar
st.write(f"Contextual Relevance Kazakh language: {context_qaz:.2%}")
st.progress(context_qaz)

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Draw a styled horizontal line
st.markdown('<hr style="border: 2px solid #00ff00; background-color: #00ff00; margin: 0px -50% 0px -50%;">', unsafe_allow_html=True)

####################################################################################################################################

# HUMAN LIKE INTERACTION
st.title("Human-like Interaction")
st.write("The Human-like Interaction metric assesses whether the responses provided by Alice exhibit qualities that resemble human-like communication. If it just gives me search result it is considered No, if it gives me answer for exactly what I asked then it is Yes.")
st.latex(r"\text{Human-like Interaction Rate} = \frac{\text{Number of Responses with Human-like Interaction}}{\text{Total Number of Responses}} \times 100")

# Whitespace
st.markdown("<br>", unsafe_allow_html=True)

# Total
hum_total = (df['Human-like interaction'] == 'Yes' ).sum()
not_hum_total = (df['Human-like interaction'] == 'No' ).sum()

human_total = hum_total / (hum_total + not_hum_total)

# Russian
hum_rus = (rus_queries['Human-like interaction'] == 'Yes' ).sum()
not_hum_rus = (rus_queries['Human-like interaction'] == 'No' ).sum()

human_rus = hum_rus / (hum_rus + not_hum_rus)

# Kazakh
hum_qaz = (qaz_queries['Human-like interaction'] == 'Yes' ).sum()
not_hum_qaz = (qaz_queries['Human-like interaction'] == 'No' ).sum()

human_qaz = hum_qaz / (hum_qaz + not_hum_qaz)

# Display Human Like Interaction
st.subheader("Human-like Interaction Progress Bars")

# Display Human-like Interaction for total using a progress bar
st.write(f"Human-like Interaction Total: {human_total:.2%}")
st.progress(human_total)

# Display Human-like Interaction for Russian using a progress bar
st.write(f"Human-like Interaction Russian language: {human_rus:.2%}")
st.progress(human_rus)

# Display Human-like Interaction for Kazakh using a progress bar
st.write(f"Human-like Interaction Kazakh language: {human_qaz:.2%}")
st.progress(human_qaz)

st.write("Interestingly enough, Kazakh version of Alice here shows a slightly better result than Russian version.")

# Save the filtered DataFrame to a new CSV file
# filtered_csv_path = 'filtered_data.csv'
# filtered_df.to_csv(filtered_csv_path, index=False)

# st.success(f"Filtered data saved to {filtered_csv_path}")




