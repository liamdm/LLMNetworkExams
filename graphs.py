"""
This script was used to generate the graphs for the paper
"""

import duckdb
import numpy as np
import pandas as pd
from nw.nw_figure import figure2d

def get_distinct(table_name, column_name):
    d = [v for (v,) in duckdb.query(f"SELECT DISTINCT({column_name}) FROM {table_name}").fetchall()]
    return d

duckdb.query("CREATE OR REPLACE TABLE results AS (SELECT * FROM read_csv('outputs/qa_df_all.csv'))")

duckdb.query("""
    CREATE OR REPLACE TABLE gpt5_1 AS
    SELECT *
    FROM read_json('outputs/gpt5.ndjson', format='nd', records=true);
""")

duckdb.query("""
    CREATE OR REPLACE TABLE gpt5_2 AS
    SELECT *
    FROM read_json('outputs/gpt5_medium.ndjson', format='nd', records=true);
""")

duckdb.query("""CREATE OR REPLACE TABLE gpt5_merged AS
WITH base AS (
    SELECT * FROM gpt5_1
    WHERE __key NOT IN (SELECT __key FROM gpt5_2)
),
overrides AS (
    SELECT * FROM gpt5_2
)
SELECT * FROM base
UNION ALL
SELECT * FROM overrides;
""")

duckdb.query("""UPDATE gpt5_merged
                SET model = model || '-r'
                WHERE __key.reasoning_effort = 'medium';
             """)

duckdb.query("""
             ALTER TABLE gpt5_merged DROP COLUMN "__key"
             """)

print("results columns:", duckdb.query("SELECT * FROM results").columns)
print("gpt5_merged columns:", duckdb.query("SELECT * FROM gpt5_merged").columns)

duckdb.query("""
             INSERT INTO results
             SELECT *
             FROM gpt5_merged
             """)

duckdb.query("SELECT DISTINCT model FROM results").show()

duckdb.query("UPDATE results SET temperature = '0.0' WHERE model = 'gpt-5.1-2025-11-13-r'")

duckdb.query("UPDATE results SET model = 'GPT-3.5 Instr.' WHERE model = 'gpt-3.5-turbo-1106'")
duckdb.query("UPDATE results SET model = 'GPT-3.5 Chat' WHERE model = 'gpt-3.5-turbo'")
duckdb.query("UPDATE results SET model = 'GPT-4o' WHERE model = 'gpt-4o'")
duckdb.query("UPDATE results SET model = 'LLama3 8B' WHERE model = 'llamaapi:llama3-8b'")
duckdb.query("UPDATE results SET model = 'GPT-5.1 Reasoning' WHERE model = 'gpt-5.1-2025-11-13-r'")
duckdb.query("UPDATE results SET model = 'GPT-5.1' WHERE model = 'gpt-5.1-2025-11-13'")
# duckdb.query("UPDATE results SET model = 'Gemma2 9B' WHERE model = 'llamaapi:gemma2-9b")

model_order = ["LLama3 8B", "GPT-3.5 Instr.", "GPT-3.5 Chat", "GPT-4o", "GPT-5.1", "GPT-5.1 Reasoning"]
model_order_dict = dict([(model_name, i) for i, model_name in enumerate(model_order)])

fig, ax = figure2d(
    legend_alpha=0.0
)

model_color_map = ["#eb4d4b", "#f0932b", "#22a6b3", "#6ab04c", "#6C5FFC", "#4D56A2"]
model_hatch_map = ["/", "+", "x", ".", "o", "*"]

model_color_map = dict([(model_order[i], model_color_map[i]) for i in range(len(model_order))])
model_hatch_map = dict([(model_order[i], model_hatch_map[i]) for i in range(len(model_order))])

g_model_color_dict = {"*": fig.colors.black}
g_model_color_dict = model_color_map
g_solid = True
fig.close()

duckdb.query("CREATE OR REPLACE TABLE tags AS (SELECT * FROM read_json('outputs/rating_df.json'))")

duckdb.query("""CREATE OR REPLACE TABLE DistinctCategories AS (
    SELECT DISTINCT UNNEST(categories) AS category FROM tags
)""")

duckdb.query("""CREATE OR REPLACE TABLE DistinctCategories AS (
    SELECT DistinctCategories.category, (SELECT COUNT(*) FROM tags WHERE list_contains(categories, DistinctCategories.category)) AS count FROM DistinctCategories WHERE count > 3 ORDER BY count DESC
)""")

duckdb.query("CREATE TABLE joined_working AS "
             "(SELECT results.*, tags.tags, tags.difficulty, tags.categories FROM results, tags "
             "WHERE results.question_id = tags.question_id "
             "AND results.exam_id = tags.exam_id "
             "AND results.working = 'yes')")

all_distinct_categories = get_distinct("DistinctCategories", "category")

models = [v for (v,) in duckdb.query("SELECT DISTINCT(model) FROM results").fetchall()]

import numpy as np


def calculate_icc(arr: np.ndarray) -> float:
    """
    Calculate the ICC
    :param arr: arr is of shape (n_questions, n_repeats) where the first dimension corresponds to the
    question and the second dimension corresponds to the repeats, with each value being
    1 if the question was answered correctly and 0 if it was answered incorrectly
    :return: the ICC for the test
    """

    # Number of questions and repeats
    n_questions, n_repeats = arr.shape

    # Calculate the mean for each question (row means) and each repeat (column means)
    row_means = np.mean(arr, axis=1)
    column_means = np.mean(arr, axis=0)

    # Calculate the grand mean
    grand_mean = np.mean(arr)

    # Calculate the total sum of squares (SS_total)
    ss_total = np.sum((arr - grand_mean) ** 2)

    # Calculate the sum of squares between questions (SS_between_rows)
    ss_between_rows = n_repeats * np.sum((row_means - grand_mean) ** 2)

    # Calculate the sum of squares within questions (SS_within_rows)
    ss_within_rows = ss_total - ss_between_rows

    # Calculate the mean squares
    ms_between_rows = ss_between_rows / (n_questions - 1)
    ms_within_rows = ss_within_rows / (n_questions * (n_repeats - 1))

    # Calculate ICC
    icc = (ms_between_rows - ms_within_rows) / (ms_between_rows + (n_repeats - 1) * ms_within_rows)

    return icc


print(duckdb.query("SELECT * FROM results").columns)

exam_abkurzung = {
    "CCNA": "CCNA",
    "CCST": "CCST",
    "Microsoft N.F.": "MicrosoftNF"
}

duckdb.query("CREATE TABLE working_results AS (SELECT * FROM results WHERE working = 'yes')")

exam_pass_score = 0.825

# ======================= CCNA results by difficulty
difficulty_categories = ["Beginner", "Intermediate", "Advanced", "Expert"]

for exam in ["CCNA"]:
    duckdb.query(f"CREATE OR REPLACE TABLE d AS (SELECT * FROM joined_working WHERE exam='{exam}')")

    all_category_df = []
    for difficulty in difficulty_categories:
        duckdb.query(f"CREATE OR REPLACE TABLE categoryD AS (SELECT * FROM d WHERE difficulty = '{difficulty}')")
        duckdb.query(
            f"CREATE OR REPLACE TABLE categoryD AS (SELECT model, repeat, SUM(CASE WHEN correct = 'yes' THEN 1 ELSE 0 END) AS n_correct, COUNT(*) AS total FROM categoryD GROUP BY model, repeat)")
        duckdb.query(
            "CREATE OR REPLACE TABLE categoryD AS (SELECT *, CAST(n_correct AS FLOAT) / total AS percent FROM categoryD)")
        df = duckdb.query(
            f"SELECT model, '{difficulty}' as category, AVG(percent) AS percent FROM categoryD GROUP BY model").df()
        all_category_df.append(df)

    all_category_df = pd.concat(all_category_df)

    # sort all_category_df on the column "model" into order by the list model_order which contains a list of the models in the order they should be in the df
    fig, ax = figure2d(legend_box=True)
    _, x_bound = ax.plot_clusteredbar(all_category_df, "model", "category", "percent", sort_l2_by_avg=True,
                                      sort_l1_by_avg=model_order_dict, return_xbounds=True,
                                      level_2_hatchmap={
                                          "Advanced": ".",
                                          "Intermediate": "+",
                                          "Beginner": "/"
                                      }, level_1_colormap=g_model_color_dict, solid=g_solid)
    ax.plot([x_bound[0], x_bound[1]], [exam_pass_score, exam_pass_score], c='black', ls='--')
    middle_x = np.mean([x_bound[0], x_bound[1]]) * 0.6
    middle_y = exam_pass_score + 0.05
    ax.text(middle_x, middle_y, f"Score > {exam_pass_score * 100:.1f}% (Passed Exam)",
            bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='none', boxstyle='round,pad=0.3'))
    ax.legend(loc='upper left')
    ax.set_ybound(0, 1)
    ax.grid(axis="y", which="major", linewidth=0.7, alpha=1.0, color="#535c68")
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Percent Correct (%)")
    fig.save(f"graphsout/ScoresByPredictedDifficulty_CCNA.pdf", width=1200, height=980, tight=True)
    fig.close()

# ======================= CCNA results by type category
type_categories = ["Conceptual Understanding", "Memory Based", "Analytical", "Calculation Based"]
type_categories_real_names = ["Conceptual Understanding", "Memory Based", "Logical Analysis", "Calculation Based"]
type_hash_map = ["///", "\\\\", "xx", "oo"]
type_hash_map = dict(zip(type_categories_real_names, type_hash_map))

for exam in ["CCNA"]:
    duckdb.query(f"CREATE OR REPLACE TABLE d AS (SELECT * FROM joined_working WHERE exam='{exam}')")

    all_category_df = []
    for question_category in type_categories:
        duckdb.query(
            f"CREATE OR REPLACE TABLE categoryD AS (SELECT * FROM d WHERE list_contains(tags, '{question_category}'))")
        duckdb.query(
            f"CREATE OR REPLACE TABLE categoryD AS (SELECT model, repeat, SUM(CASE WHEN correct = 'yes' THEN 1 ELSE 0 END) AS n_correct, COUNT(*) AS total FROM categoryD GROUP BY model, repeat)")
        duckdb.query(
            "CREATE OR REPLACE TABLE categoryD AS (SELECT *, CAST(n_correct AS FLOAT) / total AS percent FROM categoryD)")
        new_category = "Logical Analysis" if question_category == "Analytical" else question_category
        df = duckdb.query(
            f"SELECT model, '{new_category}' as category, AVG(percent) AS percent FROM categoryD GROUP BY model").df()
        all_category_df.append(df)

    all_category_df = pd.concat(all_category_df)

    fig, ax = figure2d(legend_box=True)
    _, x_bound = ax.plot_clusteredbar(all_category_df, "model", "category", "percent", sort_l2_by_avg=True,
                                      sort_l1_by_avg=model_order_dict, return_xbounds=True,
                                      level_2_hatchmap=type_hash_map, level_1_colormap=g_model_color_dict,
                                      solid=g_solid)
    #ax.plot([x_bound[0], x_bound[1]], [exam_pass_score, exam_pass_score], c='black', ls='--')
    #middle_x = np.mean([x_bound[0], x_bound[1]]) * 0.6
    #middle_y = exam_pass_score + 0.05
    #ax.text(middle_x, middle_y, f"Score > {exam_pass_score * 100:.1f}% (Passed Exam)",
    #        bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='none', boxstyle='round,pad=0.3'))
    ax.legend(loc='upper left')
    ax.set_ybound(0, 1)
    ax.grid(axis="y", which="major", linewidth=0.7, alpha=1.0, color="#535c68")
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Percent Correct (%) - By Problem Solving Type")
    fig.save(f"graphsout/ScoresByType_CCNA.pdf", width=1200, height=980, tight=True)
    fig.close()

# ======================= CCNA results by category

for exam in ["CCNA"]:
    duckdb.query(f"CREATE OR REPLACE TABLE d AS (SELECT * FROM joined_working WHERE exam='{exam}')")

    all_category_df = []
    for question_category in all_distinct_categories:
        duckdb.query(
            f"CREATE OR REPLACE TABLE categoryD AS (SELECT * FROM d WHERE list_contains(categories, '{question_category}'))")
        duckdb.query(
            f"CREATE OR REPLACE TABLE categoryD AS (SELECT model, repeat, SUM(CASE WHEN correct = 'yes' THEN 1 ELSE 0 END) AS n_correct, COUNT(*) AS total FROM categoryD GROUP BY model, repeat)")
        duckdb.query(
            "CREATE OR REPLACE TABLE categoryD AS (SELECT *, CAST(n_correct AS FLOAT) / total AS percent FROM categoryD)")
        df = duckdb.query(
            f"SELECT model, '{question_category}' as category, AVG(percent) AS percent FROM categoryD GROUP BY model").df()
        all_category_df.append(df)

    all_category_df = pd.concat(all_category_df)

    print(all_category_df)
    fig, ax = figure2d(legend_box=True, legend_font_scale=0.8)
    _, x_bound = ax.plot_clusteredbar(all_category_df, "model", "category", "percent", sort_l2_by_avg=True,
                                      sort_l1_by_avg=model_order_dict, return_xbounds=True)
    #ax.plot([x_bound[0], x_bound[1]], [exam_pass_score, exam_pass_score], c='black', ls='--')
    #middle_x = np.mean([x_bound[0], x_bound[1]]) * 0.6
    #middle_y = exam_pass_score + 0.05
    #ax.text(middle_x, middle_y, f"Score > {exam_pass_score * 100:.1f}% (Passed Exam)",
    #        bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='none', boxstyle='round,pad=0.3'))
    ax.legend(loc='upper left')
    ax.set_ybound(0, 1)
    ax.grid(axis="y", which="major", linewidth=0.7, alpha=1.0, color="#535c68")
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Percent Correct (%) - By Question Type")
    fig.save(f"graphsout/ScoresByCategory_CCNA.pdf", width=1200, height=720, tight=True)
    fig.close()

# ======================= Working vs No Working CCNA

for exam in ["CCNA", "CCST"]:
    duckdb.query(f"CREATE OR REPLACE TABLE d AS (SELECT * FROM results WHERE exam = '{exam}')")
    duckdb.query(
        "CREATE OR REPLACE TABLE d AS (SELECT model, repeat, CASE WHEN working = 'yes' THEN 'Chain Of Thought (CoT) Prompt' ELSE 'No CoT Prompt' END AS working, SUM(CASE WHEN correct = 'yes' THEN 1 ELSE 0 END) AS correct, COUNT(*) as total FROM d GROUP BY  model, repeat, working)")
    duckdb.query(
        "CREATE OR REPLACE TABLE d AS (SELECT *, CAST(correct AS float) / CAST(total AS float) AS percent FROM d)")
    df = duckdb.query("SELECT * FROM d").df()

    l = ["Chain Of Thought (CoT) Prompt", "No CoT Prompt"]
    v = ["+", "/"]
    l2_hatchmap = dict(zip(l, v))

    fig, ax = figure2d(legend_box=True)
    _, x_bound = ax.plot_clusteredbar(df, "model", "working", "percent", error_method=lambda x: (np.min(x), np.max(x)),
                                      sort_l1_by_avg=model_order_dict, return_xbounds=True, error_label="Best / Worst",
                                      level_2_hatchmap=l2_hatchmap, level_1_colormap=g_model_color_dict, solid=g_solid)
    #ax.plot([x_bound[0], x_bound[1]], [exam_pass_score, exam_pass_score], c='black', ls='--')
    #middle_x = np.mean([x_bound[0], x_bound[1]]) * 0.6
    #middle_y = exam_pass_score + 0.05
    #ax.text(middle_x, middle_y, f"Score > {exam_pass_score * 100:.1f}% (Passed Exam)",
    #        bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='none', boxstyle='round,pad=0.3'))
    ax.legend(loc='upper left')
    ax.set_ybound(0, 1)
    ax.grid(axis="y", which="major", linewidth=0.7, alpha=1.0, color="#535c68")
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xlabel("Model")
    ax.set_ylabel("Percent Correct (%)")
    fig.save(f"graphsout/WithAndWithoutWorking_{exam}.pdf", width=1000, height=720, tight=True)
    fig.close()

# ======================= Overall results

duckdb.query(
    "CREATE OR REPLACE TABLE d AS (SELECT exam, model, repeat, SUM(CASE WHEN correct = 'yes' THEN 1 ELSE 0 END) AS correct, COUNT(*) as total FROM working_results GROUP BY exam, model, repeat)")
duckdb.query("CREATE OR REPLACE TABLE d AS (SELECT *, CAST(correct AS float) / CAST(total AS float) AS percent FROM d)")
df = duckdb.query("SELECT * FROM d").df()

fig, ax = figure2d(legend_box=True, legend_font_scale=0.75)
_, x_bound = ax.plot_clusteredbar(
    df,
    "exam", "model", "percent",
    error_method=lambda x: (np.min(x), np.max(x)),
    return_xbounds=True,
    sort_l2_by_avg=model_order_dict,
    error_label="Best / Worst",
    level_2_colormap=g_model_color_dict
)
ax.plot([x_bound[0], x_bound[1]], [exam_pass_score, exam_pass_score], c='black', ls='--')
middle_x = np.mean([x_bound[0], x_bound[1]]) * 0.6
middle_y = exam_pass_score + 0.05
ax.text(middle_x, middle_y, f"Score > {exam_pass_score * 100:.1f}% (Passed Exam)",
        bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='none', boxstyle='round,pad=0.3'))
ax.legend(loc='upper left')
ax.set_ybound(0, 1)
ax.grid(axis="y", which="major", linewidth=0.7, alpha=1.0, color="#535c68")
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.set_xlabel("Exam")
ax.set_ylabel("Percent Correct (%)")
fig.save(f"graphsout/OverallExamScores.pdf", width=1200, height=720, tight=True)
fig.close()

# # ======================= Reliability confusion matrix

duckdb.query(
    "CREATE OR REPLACE TABLE d AS (SELECT exam, model, repeat, SUM(CASE WHEN correct = 'yes' THEN 1 ELSE 0 END) AS correct, COUNT(*) as total FROM working_results GROUP BY exam, model, repeat)")
duckdb.query("CREATE OR REPLACE TABLE d AS (SELECT *, CAST(correct AS float) / CAST(total AS float) AS percent FROM d)")

exams = get_distinct("working_results", "exam")
models = get_distinct("working_results", "model")

for exam in exams:
    output_script = f"%% ~~~ Tables for {exam} ~~~\n"

    for model in models:
        duckdb.query(
            f"CREATE OR REPLACE TABLE d AS (SELECT * FROM joined_working WHERE exam = '{exam}' AND model = '{model}' AND temperature = 0.2 AND difficulty='Intermediate')")

        # if model == "gpt-3.5-turbo-1106":
        #    continue

        # it gets a question consistently correct if all repeats are 1
        # consistently incorrect if all repeats are 0

        number_questions = duckdb.query("SELECT COUNT(DISTINCT(question_id)) FROM d").fetchone()[0]

        if number_questions == 0:
            print(f'Skipping @ {model} - no answered questions!')
            continue

        number_questions_got_correct = duckdb.query(
            "CREATE OR REPLACE TABLE gc AS (SELECT question_id, (SELECT COUNT(*) FROM d WHERE correct = 'yes' AND d.question_id = X.question_id) AS got_correct, (SELECT COUNT(*) FROM d WHERE correct != 'yes' AND d.question_id = X.question_id) AS got_incorrect FROM d as X GROUP BY X.question_id)")
        duckdb.query("SELECT * FROM gc").show()

        n_ever_correct = duckdb.query("SELECT COUNT(*) FROM gc WHERE got_correct > 0 AND got_correct < 5").fetchone()[0]
        n_ever_incorrect = \
            duckdb.query("SELECT COUNT(*) FROM gc WHERE got_incorrect > 0 AND got_incorrect < 5").fetchone()[0]
        n_always_correct = duckdb.query("SELECT COUNT(*) FROM gc WHERE got_correct == 5").fetchone()[0]
        n_always_incorrect = duckdb.query("SELECT COUNT(*) FROM gc WHERE got_incorrect == 5").fetchone()[0]

        total_consistent = n_always_correct + n_always_incorrect
        total_inconsistent = n_ever_correct

        total_consistent_percent = total_consistent / number_questions
        total_inconsistent_percent = total_inconsistent / number_questions

        print("n_ever_correct", n_ever_correct, n_ever_correct / number_questions)
        print("n_ever_incorrect", n_ever_incorrect, n_ever_incorrect / number_questions)
        print("n_always_correct", n_always_correct, n_always_correct / number_questions)
        print("n_always_incorrect", n_always_incorrect, n_always_incorrect / number_questions)

        #
        template = r"""
\begin{table}[]
\centering
\resizebox{0.9\columnwidth}{!}{%
\begin{tabular}{@{}cccl@{}}
\toprule
\multicolumn{1}{l}{%n%}                                            & \begin{tabular}[c]{@{}c@{}}Answered\\ Correctly\end{tabular} & \begin{tabular}[c]{@{}c@{}}Answered\\ Incorrectly\end{tabular} & \textbf{Total} \\ \midrule
\begin{tabular}[c]{@{}c@{}}Answered\\ Consistently\end{tabular}    & %n_always_correct% (%n_always_correct_percent%)  & %n_always_incorrect% (%n_always_incorrect_percent%)    & %total_consistent% (%total_consistent_percent%)                                                        \\
\begin{tabular}[c]{@{}c@{}}Answered \\ Inconsistently\end{tabular} & \multicolumn{2}{c}{%n_ever_correct% (%n_ever_correct_percent%)}  & %total_inconsistent% (%total_inconsistent_percent%)      \\ \bottomrule
\end{tabular}%
}
\caption{%model% consistency matrix for the %dataset% question set}
\label{tab:%model%_%dataset%_ConfusionMatrix}
\end{table}"""

        n_ever_correct_percent = n_ever_correct / number_questions
        n_ever_incorrect_percent = n_ever_incorrect / number_questions
        n_always_correct_percent = n_always_correct / number_questions
        n_always_incorrect_percent = n_always_incorrect / number_questions

        template = template.replace("%n_always_correct%", f"{n_always_correct}")
        template = template.replace("%n_always_incorrect%", f"{n_always_incorrect}")
        template = template.replace("%n_ever_correct%", f"{n_ever_correct}")
        template = template.replace("%n_ever_incorrect%", f"{n_ever_incorrect}")

        template = template.replace("%total_consistent%", f"{total_consistent}")
        template = template.replace("%total_inconsistent%", f"{total_inconsistent}")
        template = template.replace("%total_consistent_percent%", f"{total_consistent_percent * 100:.2f}\\%")
        template = template.replace("%total_inconsistent_percent%", f"{total_inconsistent_percent * 100:.2f}\\%")

        template = template.replace("%n%", f"N = {number_questions}")

        template = template.replace("%n_ever_correct_percent%", f"{n_ever_correct_percent * 100:.2f}\\%")
        template = template.replace("%n_ever_incorrect_percent%", f"{n_ever_incorrect_percent * 100:.2f}\\%")
        template = template.replace("%n_always_correct_percent%", f"{n_always_correct_percent * 100:.2f}\\%")
        template = template.replace("%n_always_incorrect_percent%", f"{n_always_incorrect_percent * 100:.2f}\\%")

        output_script += f"% ==== %model%\n{template}".replace("%model%", model).replace("%dataset%",
                                                                                         exam_abkurzung[exam])

    with open(f"graphsout/{exam_abkurzung[exam]}_confusion.tex", "w") as w:
        w.write(output_script)

# ======================= Temperature vs correctness
duckdb.query(
    "CREATE TABLE temp_v_correctness AS (SELECT exam, model, repeat, temperature, SUM(CASE WHEN correct = 'yes' THEN 1 ELSE 0 END) AS n_correct, COUNT(*) AS n_overall, CAST(n_correct AS FLOAT) / CAST(n_overall AS FLOAT) AS percent FROM working_results AS X GROUP BY exam, model, repeat, temperature)")
duckdb.query("SELECT * FROM temp_v_correctness").show()

exams = get_distinct("temp_v_correctness", "exam")
for exam in exams:
    df = duckdb.query(f"SELECT * FROM temp_v_correctness WHERE exam = '{exam}'").df()
    df['temperature'] = df['temperature'].replace({0.2: 't=0.2', 0.7: 't=0.7', 0.0: 'N/A'})

    l2_hatchmap = dict(zip(['t=0.2', 't=0.7', 'N/A'], ['o', 'x', '/']))

    fig, ax = figure2d(legend_box=True)
    ax.plot_clusteredbar(df, "model", "temperature", "percent", error_method=lambda x: (np.min(x), np.max(x)),
                         error_label="Best / Worst Repeat", sort_l1_by_avg=model_order_dict,
                         level_2_hatchmap=l2_hatchmap, level_1_colormap=g_model_color_dict, solid=g_solid)
    ax.set_xlabel("Model")
    ax.set_ylabel("Percent Correct (%)")
    ax.set_ybound(0, 1)
    ax.grid(axis="y", which="major", linewidth=0.7, alpha=1.0, color="#535c68")
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    fig.save(f"graphsout/{exam_abkurzung[exam]}_TempVsPerformance.pdf", width=980, height=720, tight=True)
    fig.close()

# ======================= Reliability overall

# we want a table that shows each model, temperature and ICC
duckdb.query("CREATE TABLE icc_table AS (SELECT * FROM results WHERE working = 'yes')")

exams = get_distinct("icc_table", "exam_id")
temperatures = get_distinct("icc_table", "temperature")
workings = get_distinct("icc_table", "working")

duckdb.query('''
             CREATE TABLE transformed_table AS
             SELECT model,
                    exam,
                    temperature,
                    question_id,
                    MAX(CASE WHEN repeat = 1 THEN correct = 'yes' ELSE 0 END) AS repeat_1_correct,
                    MAX(CASE WHEN repeat = 2 THEN correct = 'yes' ELSE 0 END) AS repeat_2_correct,
                    MAX(CASE WHEN repeat = 3 THEN correct = 'yes' ELSE 0 END) AS repeat_3_correct,
                    MAX(CASE WHEN repeat = 4 THEN correct = 'yes' ELSE 0 END) AS repeat_4_correct,
                    MAX(CASE WHEN repeat = 5 THEN correct = 'yes' ELSE 0 END) AS repeat_5_correct
             FROM icc_table
             GROUP BY model, exam, temperature, question_id
             ''')

exams = get_distinct("transformed_table", "exam")
for exam in exams:
    transformed_df = duckdb.query(f"SELECT * FROM transformed_table WHERE exam = '{exam}'").df()
    grouped = transformed_df.groupby(['model', 'exam', 'temperature'])

    exam_results = []
    # Iterate through each group and calculate ICC
    for (model, exam_id, temperature), group in grouped:
        # Create the numpy array of shape (n_questions, n_repeats)
        questions_array = group[
            ['repeat_1_correct', 'repeat_2_correct', 'repeat_3_correct', 'repeat_4_correct', 'repeat_5_correct']].values

        # Replace NaN with 0 (assuming NaN means the question was not answered correctly)
        questions_array = np.nan_to_num(questions_array)

        # Calculate ICC
        icc = calculate_icc(questions_array)

        # Store the ICC result
        exam_results.append({
            "Model": model,
            "Exam": exam,
            "Temperature": f"t={temperature}" if temperature > 0.0 else "N/A",
            "ICC": icc
        })

    l2_hatchmap = dict(zip(['t=0.2', 't=0.7', 'N/A'], ['o', 'x', '/']))

    exam_results = pd.DataFrame(exam_results)
    fig, ax = figure2d(legend_box=True)
    ax.legend_label("Temperature")
    _, x_bound = ax.plot_clusteredbar(exam_results, "Model", "Temperature", "ICC", sort_l1_by_avg=model_order_dict,
                                      return_xbounds=True,
                                      level_2_hatchmap=l2_hatchmap, level_1_colormap=g_model_color_dict, solid=g_solid)
    ax.plot([x_bound[0], x_bound[1]], [0.8, 0.8], c='black', ls='--')
    middle_x = np.mean([x_bound[0], x_bound[1]]) * 0.6
    middle_y = 0.8
    ax.text(middle_x, middle_y, "ICC > 0.8 (Reliable)",
            bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='none', boxstyle='round,pad=0.3'))
    ax.legend()
    ax.set_ybound(0, 1)
    ax.grid(axis="y", which="major", linewidth=0.7, alpha=1.0, color="#535c68")
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    fig.save(f"graphsout/{exam_abkurzung[exam]}_Reliability.pdf", width=980, height=720, tight=True)
    fig.close()
