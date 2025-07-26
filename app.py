import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Models Comparison", "Occupation-Based Recommendation"])

if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Pie chart data
    labels = ['Accepted', 'Rejected']
    sizes = [56.8, 43.2]
    colors = ['#90ee90', '#f98484']

    # Adjusted figure size and font sizes
    fig1, ax1 = plt.subplots(figsize=(2.5, 2.5))  # compact but readable
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 8}  # balanced label size
    )
    ax1.axis('equal')  # Keep the pie circular

    # Subtitle smaller than main title
    ax1.set_title("Overall Coupon Acceptance Rate", fontsize=10, pad=10)

    # Display in Streamlit
    st.pyplot(fig1)






    # Main EDA Sections as Tabs
    main_tabs = st.tabs(["Coupon Type", "Demographics", "Contextual"])

    # --------------------- Coupon Type Tab ---------------------
    with main_tabs[0]:
        st.subheader("Coupon Acceptance by Type")
        data = {
            "Accepted %": [41.0, 73.5, 49.9, 44.1, 70.7],
            "Rejected %": [59.0, 26.5, 50.1, 55.9, 29.3]
        }
        index = ["Bar", "Fast Food", "Café", "Inexpensive restaurant", "Mid-range restaurant"]
        grouped = pd.DataFrame(data, index=index)

        # Plot
        colors = ['#f98484', '#90ee90']
        fig, ax = plt.subplots(figsize=(12, 4))
        y = range(len(grouped.index))
        ax.barh(y, grouped['Accepted %'], color=colors[1], label='Accepted %')
        ax.barh(y, grouped['Rejected %'], left=grouped['Accepted %'], color=colors[0], label='Rejected %')
        for i, (accept, reject) in enumerate(zip(grouped['Accepted %'], grouped['Rejected %'])):
            ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=10)
            ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=10)
        ax.set_title("Accepted vs. Rejected Coupons by Coupon Type")
        ax.set_xlabel("Percentage")
        ax.set_xlim(0, 100)
        ax.set_yticks(y)
        ax.set_yticklabels(grouped.index)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig)

    # --------------------- Demographics Tab ---------------------
    with main_tabs[1]:
        st.subheader("Demographic Analysis")
        demo_tabs = st.tabs(["Occupation", "Education", "Age Group", "Income Group", "Gender", "Marital Group"])

        # Occupation (Placeholder)
        # Occupation Group (Your real data)
        with demo_tabs[0]:
            st.write("Analysis by **Occupation**")

            occupation_data = {
                "Accepted %": [
                    63.43, 51.83, 59.09, 56.99, 48.96, 56.68, 68.83, 52.39, 53.49,
                    58.39, 67.62, 69.83, 53.38, 47.03, 57.65, 58.83, 60.09, 54.86,
                    61.82, 64.57, 45.86, 56.27, 61.05, 59.63, 54.81
                ],
                "Rejected %": [
                    36.57, 48.17, 40.91, 43.01, 51.04, 43.32, 31.17, 47.61, 46.51,
                    41.61, 32.38, 30.17, 46.62, 52.97, 42.35, 41.17, 39.91, 45.14,
                    38.18, 35.43, 54.14, 43.73, 38.95, 40.37, 45.19
                ]
            }
            occupation_index = [
                "Architecture & Engineering",
                "Arts Design Entertainment Sports & Media",
                "Building & Grounds Cleaning & Maintenance",
                "Business & Financial",
                "Community & Social Services",
                "Computer & Mathematical",
                "Construction & Extraction",
                "Education&Training&Library",
                "Farming Fishing & Forestry",
                "Food Preparation & Serving Related",
                "Healthcare Practitioners & Technical",
                "Healthcare Support",
                "Installation Maintenance & Repair",
                "Legal",
                "Life Physical Social Science",
                "Management",
                "Office & Administrative Support",
                "Personal Care & Service",
                "Production Occupations",
                "Protective Service",
                "Retired",
                "Sales & Related",
                "Student",
                "Transportation & Material Moving",
                "Unemployed"
            ]

            occupation_df = pd.DataFrame(occupation_data, index=occupation_index)

            fig, ax = plt.subplots(figsize=(12, 8))
            y = range(len(occupation_df.index))
            ax.barh(y, occupation_df['Accepted %'], color='#90ee90', label='Accepted %')
            ax.barh(y, occupation_df['Rejected %'], left=occupation_df['Accepted %'], color='#f98484', label='Rejected %')

            for i, (accept, reject) in enumerate(zip(occupation_df['Accepted %'], occupation_df['Rejected %'])):
                ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=7)
                ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=7)

            ax.set_title("Coupon Acceptance by Occupation")
            ax.set_xlabel("Percentage")
            ax.set_xlim(0, 100)
            ax.set_yticks(y)
            ax.set_yticklabels(occupation_df.index)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            st.pyplot(fig)


        # Education (Your real data)
        with demo_tabs[1]:
            st.write("Analysis by **Education**")

            edu_data = {
                "Accepted %": [55.3, 55.4, 52.6, 59.2, 71.6, 59.6],
                "Rejected %": [44.7, 44.6, 47.4, 40.8, 28.4, 40.4]
            }
            edu_index = [
                "Associates degree",
                "Bachelors degree",
                "Graduate degree (Masters or Doctorate)",
                "High School",
                "Some High School",
                "Some college - no degree"
            ]
            edu_df = pd.DataFrame(edu_data, index=edu_index)

            fig, ax = plt.subplots(figsize=(10, 5))
            y = range(len(edu_df.index))
            ax.barh(y, edu_df['Accepted %'], color='#90ee90', label='Accepted %')
            ax.barh(y, edu_df['Rejected %'], left=edu_df['Accepted %'], color='#f98484', label='Rejected %')

            for i, (accept, reject) in enumerate(zip(edu_df['Accepted %'], edu_df['Rejected %'])):
                ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=9)
                ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=9)

            ax.set_title("Coupon Acceptance by Education Level")
            ax.set_xlabel("Percentage")
            ax.set_xlim(0, 100)
            ax.set_yticks(y)
            ax.set_yticklabels(edu_df.index)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)

        # Age Group (Your real data + explanation)
        with demo_tabs[2]:
            st.write("Analysis by **Age Group**")

            age_data = {
                "Accepted %": [56.53, 57.39, 50.89, 60.44],
                "Rejected %": [43.47, 42.61, 49.11, 39.56]
            }
            age_index = ["Adult", "MiddleAged", "Senior", "Young"]
            age_df = pd.DataFrame(age_data, index=age_index)

            fig, ax = plt.subplots(figsize=(8, 4))
            y = range(len(age_df.index))
            ax.barh(y, age_df['Accepted %'], color='#90ee90', label='Accepted %')
            ax.barh(y, age_df['Rejected %'], left=age_df['Accepted %'], color='#f98484', label='Rejected %')

            for i, (accept, reject) in enumerate(zip(age_df['Accepted %'], age_df['Rejected %'])):
                ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=9)
                ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=9)

            ax.set_title("Coupon Acceptance by Age Group")
            ax.set_xlabel("Percentage")
            ax.set_xlim(0, 100)
            ax.set_yticks(y)
            ax.set_yticklabels(age_df.index)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)

            # Age group explanation
            st.markdown("**Age Group Definitions:**")
            st.markdown("""
            - **Young**: aged 18–25 years
            - **Adult**: aged 26–35 years
            - **MiddleAged**: aged 36–50 years
            - **Senior**: aged 51+ years
            """)


        # Income Group (Your real data + explanation)
        with demo_tabs[3]:
            st.write("Analysis by **Income Group**")

            income_data = {
                "Accepted %": [54.30, 58.62, 56.89],
                "Rejected %": [45.70, 41.38, 43.11]
            }
            income_index = ["High", "Low", "Medium"]
            income_df = pd.DataFrame(income_data, index=income_index)

            fig, ax = plt.subplots(figsize=(8, 4))
            y = range(len(income_df.index))
            ax.barh(y, income_df['Accepted %'], color='#90ee90', label='Accepted %')
            ax.barh(y, income_df['Rejected %'], left=income_df['Accepted %'], color='#f98484', label='Rejected %')

            for i, (accept, reject) in enumerate(zip(income_df['Accepted %'], income_df['Rejected %'])):
                ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=9)
                ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=9)

            ax.set_title("Coupon Acceptance by Income Group")
            ax.set_xlabel("Percentage")
            ax.set_xlim(0, 100)
            ax.set_yticks(y)
            ax.set_yticklabels(income_df.index)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)

            # Income group explanation
            st.markdown("**Income Group Definitions:**")
            st.markdown("""
            - **Low** income: below $37,499
            - **Medium** income: between $37,499 and $74,999
            - **High** income: $75,000 and above
            """)


        # Gender (Placeholder)
        # Gender Group
        with demo_tabs[4]:
            st.write("Analysis by **Gender**")

            gender_data = {
                "Accepted %": [54.7, 59.1],
                "Rejected %": [45.3, 40.9]
            }
            gender_index = ["Female", "Male"]
            gender_df = pd.DataFrame(gender_data, index=gender_index)

            fig, ax = plt.subplots(figsize=(6, 3))
            y = range(len(gender_df.index))
            ax.barh(y, gender_df['Accepted %'], color='#90ee90', label='Accepted %')
            ax.barh(y, gender_df['Rejected %'], left=gender_df['Accepted %'], color='#f98484', label='Rejected %')

            for i, (accept, reject) in enumerate(zip(gender_df['Accepted %'], gender_df['Rejected %'])):
                ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=10)
                ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=10)

            ax.set_title("Coupon Acceptance by Gender")
            ax.set_xlabel("Percentage")
            ax.set_xlim(0, 100)
            ax.set_yticks(y)
            ax.set_yticklabels(gender_df.index)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)


        # Marital Group (Placeholder)
        # Marital Group (Your real data)
        with demo_tabs[5]:
            st.write("Analysis by **Marital Group**")

            marital_data = {
                "Accepted %": [54.8, 59.5],
                "Rejected %": [45.2, 40.5]
            }
            marital_index = ["Partnered", "Single"]
            marital_df = pd.DataFrame(marital_data, index=marital_index)

            fig, ax = plt.subplots(figsize=(6, 3))
            y = range(len(marital_df.index))
            ax.barh(y, marital_df['Accepted %'], color='#90ee90', label='Accepted %')
            ax.barh(y, marital_df['Rejected %'], left=marital_df['Accepted %'], color='#f98484', label='Rejected %')

            for i, (accept, reject) in enumerate(zip(marital_df['Accepted %'], marital_df['Rejected %'])):
                ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=10)
                ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=10)

            ax.set_title("Coupon Acceptance by Marital Group")
            ax.set_xlabel("Percentage")
            ax.set_xlim(0, 100)
            ax.set_yticks(y)
            ax.set_yticklabels(marital_df.index)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)

    # --------------------- Contextual Tab ---------------------
    with main_tabs[2]:
        st.subheader("Contextual Analysis")
        context_tabs = st.tabs(["Weather", "Time of Day"])

        # Weather Analysis
        with context_tabs[0]:
            st.write("Analysis by **Weather**")

            weather_data = {
                "Accepted %": [46.3, 47.0, 59.5],
                "Rejected %": [53.7, 53.0, 40.5]
            }
            weather_index = ["Rainy", "Snowy", "Sunny"]
            weather_df = pd.DataFrame(weather_data, index=weather_index)

            fig, ax = plt.subplots(figsize=(6, 3))
            y = range(len(weather_df.index))
            ax.barh(y, weather_df['Accepted %'], color='#90ee90', label='Accepted %')
            ax.barh(y, weather_df['Rejected %'], left=weather_df['Accepted %'], color='#f98484', label='Rejected %')

            for i, (accept, reject) in enumerate(zip(weather_df['Accepted %'], weather_df['Rejected %'])):
                ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=10)
                ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=10)

            ax.set_title("Coupon Acceptance by Weather")
            ax.set_xlabel("Percentage")
            ax.set_xlim(0, 100)
            ax.set_yticks(y)
            ax.set_yticklabels(weather_df.index)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)


        # Time of Day Analysis
        with context_tabs[1]:
            st.write("Analysis by **Time of Day**")

            time_data = {
                "Accepted %": [66.2, 55.5, 54.7],
                "Rejected %": [33.8, 44.5, 45.3]
            }
            time_index = ["Afternoon", "Evening", "Morning"]
            time_df = pd.DataFrame(time_data, index=time_index)

            fig, ax = plt.subplots(figsize=(6, 3))
            y = range(len(time_df.index))
            ax.barh(y, time_df['Accepted %'], color='#90ee90', label='Accepted %')
            ax.barh(y, time_df['Rejected %'], left=time_df['Accepted %'], color='#f98484', label='Rejected %')

            for i, (accept, reject) in enumerate(zip(time_df['Accepted %'], time_df['Rejected %'])):
                ax.text(accept / 2, i, f'{accept:.1f}%', va='center', ha='center', fontsize=10)
                ax.text(accept + reject / 2, i, f'{reject:.1f}%', va='center', ha='center', fontsize=10)

            ax.set_title("Coupon Acceptance by Time of Day")
            ax.set_xlabel("Percentage")
            ax.set_xlim(0, 100)
            ax.set_yticks(y)
            ax.set_yticklabels(time_df.index)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)


# --------------------- Models Comparison Page ---------------------
elif page == "Models Comparison":
    st.title("Models Comparison")

    # Main tabs: Before and After Improvement
    before_tab, after_tab = st.tabs(["Before Improvement", "After Improvement"])

    # --------- BEFORE IMPROVEMENT ---------
    with before_tab:
        st.subheader("Before Model Improvement")

        metrics_tab, time_tab = st.tabs(["Evaluation Metrics", "Training Time"])

        with metrics_tab:
            # Evaluation Metrics Before
            data = {
                'Model': ['CatBoost', 'DNN', 'DeepFM'],
                'Accuracy': [0.694915, 0.667324, 0.716200],
                'AUC': [0.755964, 0.721681, 0.761810],
                'Precision': [0.689326, 0.707568, 0.715987],
                'Recall': [0.820440, 0.683463, 0.810504],
                'F1': [0.749190, 0.695307, 0.760320],
                'Time(s)': [0.213689, 47.525431, 76.112175]
            }
            results = pd.DataFrame(data)
            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']
            models = results['Model'].tolist()
            metric_values = {metric: results[metric].values for metric in metrics}
            colors = {'CatBoost': 'skyblue', 'DNN': 'orange', 'DeepFM': 'lightgreen'}

            x = np.arange(len(metrics))
            width = 0.25
            fig, ax = plt.subplots(figsize=(10, 4))
            for i, model in enumerate(models):
                scores = [metric_values[m][i] for m in metrics]
                bars = ax.bar(x + i * width, scores, width, label=model, color=colors[model])
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.3f}',
                            ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x + width)
            ax.set_xticklabels(metrics)
            ax.set_ylabel('Score')
            ax.set_title('Evaluation Metrics Comparison (Before)')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.grid(False)
            plt.ylim(0, 1)
            st.pyplot(fig)

        with time_tab:
            # Training Time Before
            time_values = results['Time(s)'].values
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(models, time_values, color=[colors[m] for m in models])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.3f}s',
                        ha='center', va='bottom', fontsize=8)
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Time Comparison (Before)')
            ax.grid(False)
            st.pyplot(fig)

    # --------- AFTER IMPROVEMENT ---------
    with after_tab:
        st.subheader("After Model Improvement")

        metrics_tab, time_tab = st.tabs(["Evaluation Metrics", "Training Time"])

        with metrics_tab:
            # Evaluation Metrics After
            data = {
                'Model': ['CatBoost', 'DNN', 'DeepFM'],
                'Accuracy': [0.754, 0.693, 0.773],
                'AUC': [0.832, 0.758, 0.841],
                'Precision': [0.803, 0.716, 0.812],
                'Recall': [0.784, 0.740, 0.804],
                'F1': [0.784, 0.728, 0.805],
                'Time(s)': [16.005314, 58.163980, 101.224504]
            }
            results = pd.DataFrame(data)
            results['F1'].fillna((results['Precision'] + results['Recall']) / 2, inplace=True)
            metric_values = {metric: results[metric].values for metric in metrics}
            time_values = results['Time(s)'].values

            fig, ax = plt.subplots(figsize=(10, 4))
            for i, model in enumerate(models):
                scores = [metric_values[m][i] for m in metrics]
                bars = ax.bar(x + i * width, scores, width, label=model, color=colors[model])
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.3f}',
                            ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x + width)
            ax.set_xticklabels(metrics)
            ax.set_ylabel('Score')
            ax.set_title('Evaluation Metrics Comparison (After)')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.grid(False)
            plt.ylim(0, 1)
            st.pyplot(fig)

        with time_tab:
            # Training Time After
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(models, time_values, color=[colors[m] for m in models])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.3f}s',
                        ha='center', va='bottom', fontsize=8)
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Time Comparison (After)')
            ax.grid(False)
            st.pyplot(fig)


# --------------------- Occupation-Based Recommendation ---------------------
elif page == "Occupation-Based Recommendation":
    st.title("Occupation-Based Recommendation")

    main_tab1, main_tab2, main_tab3 = st.tabs(["Coupon Type", "Demographics", "Contextual"])

    # Helper function to plot occupation probabilities horizontally with colors
    def plot_occupation_probs(title, data):
        df = pd.DataFrame(data, columns=["Occupation", "Avg Predicted Probability"])
        df = df.sort_values(by="Avg Predicted Probability")

        colors = []
        for val in df["Avg Predicted Probability"]:
            if val >= 0.75:
                colors.append('green')
            elif val > 0.5:
                colors.append('yellow')
            else:
                colors.append('red')

        plt.figure(figsize=(12, max(6, len(df) * 0.3)))
        ax = plt.gca()
        bars = ax.barh(df["Occupation"], df["Avg Predicted Probability"], color=colors)
        plt.title(f'DeepFM Predicted Probability by Occupation (Coupon Type: {title})')
        plt.xlabel('Average Predicted Probability')
        plt.xlim(0, 1)

        # Add labels on bars
        for bar, val in zip(bars, df["Avg Predicted Probability"]):
            width = bar.get_width()
            yloc = bar.get_y() + bar.get_height() / 2
            if width + 0.05 > 1.0:
                ax.text(width - 0.03, yloc, f'{val:.3f}', va='center', ha='right', fontsize=9, color='black')
            else:
                ax.text(width + 0.01, yloc, f'{val:.3f}', va='center', ha='left', fontsize=9)


        green_patch = mpatches.Patch(color='green', label='Highly Recommended (≥ 0.75)')
        yellow_patch = mpatches.Patch(color='yellow', label='Recommended (0.5 - 0.75)')
        red_patch = mpatches.Patch(color='red', label='Not Recommended (≤ 0.5)')
        plt.legend(handles=[green_patch, yellow_patch, red_patch], loc='lower right')
        plt.legend(handles=[green_patch, yellow_patch, red_patch],loc='center left',bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()  # Clear figure after plotting to avoid overlap

    with main_tab1:
        st.subheader("By Coupon Type")

        # Your fixed occupation-probability data for each coupon type:
        coupon_dict = {
            "Coffee House": [
                ("Healthcare Practitioners & Technical", 0.837),
                ("Installation Maintenance & Repair", 0.757),
                ("Healthcare Support", 0.725),
                ("Legal", 0.644),
                ("Transportation & Material Moving", 0.632),
                ("Construction & Extraction", 0.596),
                ("Unemployed", 0.594),
                ("Food Preparation & Serving Related", 0.578),
                ("Student", 0.564),
                ("Retired", 0.545),
                ("Management", 0.541),
                ("Business & Financial", 0.524),
                ("Architecture & Engineering", 0.518),
                ("Computer & Mathematical", 0.502),
                ("Arts Design Entertainment Sports & Media", 0.492),
                ("Life Physical Social Science", 0.490),
                ("Office & Administrative Support", 0.478),
                ("Education&Training&Library", 0.470),
                ("Personal Care & Service", 0.455),
                ("Sales & Related", 0.418),
                ("Production Occupations", 0.417),
                ("Protective Service", 0.410),
                ("Farming Fishing & Forestry", 0.401),
                ("Community & Social Services", 0.343),
                ("Building & Grounds Cleaning & Maintenance", 0.004),
            ],
            "Bar": [
                ("Construction & Extraction", 0.879),
                ("Installation Maintenance & Repair", 0.643),
                ("Healthcare Practitioners & Technical", 0.625),
                ("Healthcare Support", 0.617),
                ("Protective Service", 0.594),
                ("Production Occupations", 0.580),
                ("Architecture & Engineering", 0.560),
                ("Office & Administrative Support", 0.539),
                ("Management", 0.522),
                ("Business & Financial", 0.516),
                ("Farming Fishing & Forestry", 0.502),
                ("Life Physical Social Science", 0.494),
                ("Food Preparation & Serving Related", 0.461),
                ("Sales & Related", 0.452),
                ("Student", 0.436),
                ("Transportation & Material Moving", 0.357),
                ("Education&Training&Library", 0.314),
                ("Computer & Mathematical", 0.293),
                ("Unemployed", 0.281),
                ("Arts Design Entertainment Sports & Media", 0.267),
                ("Personal Care & Service", 0.264),
                ("Community & Social Services", 0.211),
                ("Legal", 0.159),
                ("Retired", 0.102),
                ("Building & Grounds Cleaning & Maintenance", 0.009),
            ],
            # Add the other coupon types similarly ...
            "Restaurant (<$20)": [
                ("Installation Maintenance & Repair", 0.976),
                ("Protective Service", 0.898),
                ("Construction & Extraction", 0.860),
                ("Management", 0.853),
                ("Legal", 0.806),
                ("Life Physical Social Science", 0.791),
                ("Architecture & Engineering", 0.786),
                ("Transportation & Material Moving", 0.742),
                ("Sales & Related", 0.738),
                ("Office & Administrative Support", 0.732),
                ("Education&Training&Library", 0.727),
                ("Student", 0.726),
                ("Computer & Mathematical", 0.724),
                ("Healthcare Support", 0.714),
                ("Business & Financial", 0.708),
                ("Unemployed", 0.668),
                ("Community & Social Services", 0.642),
                ("Arts Design Entertainment Sports & Media", 0.625),
                ("Personal Care & Service", 0.618),
                ("Farming Fishing & Forestry", 0.590),
                ("Production Occupations", 0.584),
                ("Healthcare Practitioners & Technical", 0.577),
                ("Retired", 0.548),
                ("Food Preparation & Serving Related", 0.533),
            ],
            "Restaurant ($20-50)": [
                ("Legal", 0.861),
                ("Construction & Extraction", 0.818),
                ("Protective Service", 0.765),
                ("Healthcare Support", 0.759),
                ("Community & Social Services", 0.680),
                ("Management", 0.650),
                ("Office & Administrative Support", 0.641),
                ("Business & Financial", 0.637),
                ("Food Preparation & Serving Related", 0.610),
                ("Computer & Mathematical", 0.570),
                ("Education&Training&Library", 0.505),
                ("Arts Design Entertainment Sports & Media", 0.483),
                ("Architecture & Engineering", 0.462),
                ("Sales & Related", 0.461),
                ("Unemployed", 0.431),
                ("Healthcare Practitioners & Technical", 0.418),
                ("Student", 0.402),
                ("Installation Maintenance & Repair", 0.292),
                ("Personal Care & Service", 0.249),
                ("Life Physical Social Science", 0.220),
                ("Transportation & Material Moving", 0.217),
                ("Farming Fishing & Forestry", 0.164),
                ("Production Occupations", 0.136),
                ("Retired", 0.113),
                ("Building & Grounds Cleaning & Maintenance", 0.022),
            ],
            "Carry out & Take away": [
                ("Building & Grounds Cleaning & Maintenance", 0.996),
                ("Construction & Extraction", 0.983),
                ("Protective Service", 0.976),
                ("Food Preparation & Serving Related", 0.895),
                ("Transportation & Material Moving", 0.877),
                ("Office & Administrative Support", 0.833),
                ("Business & Financial", 0.790),
                ("Management", 0.771),
                ("Farming Fishing & Forestry", 0.764),
                ("Healthcare Practitioners & Technical", 0.744),
                ("Unemployed", 0.733),
                ("Healthcare Support", 0.724),
                ("Computer & Mathematical", 0.686),
                ("Education&Training&Library", 0.686),
                ("Personal Care & Service", 0.684),
                ("Retired", 0.681),
                ("Sales & Related", 0.680),
                ("Installation Maintenance & Repair", 0.666),
                ("Student", 0.647),
                ("Life Physical Social Science", 0.615),
                ("Community & Social Services", 0.614),
                ("Arts Design Entertainment Sports & Media", 0.609),
                ("Legal", 0.605),
                ("Production Occupations", 0.516),
                ("Architecture & Engineering", 0.416),
            ],
        }

        # Create tabs for each coupon type with the plots inside



        # Add 'All (Comparison)' as the first tab
        tab_names = ["All (Comparison)"] + list(coupon_dict.keys())
        coupon_tabs = st.tabs(tab_names)

        for tab, tab_name in zip(coupon_tabs, tab_names):
            with tab:
                if tab_name == "All (Comparison)":
                    st.subheader("Comparison Across All Coupon Types by Occupation")

                    # Flatten coupon_dict into a list of rows
                    all_data = []
                    for coupon_type, data in coupon_dict.items():
                        for occupation, prob in data:
                            all_data.append({
                                "Occupation": occupation,
                                "Coupon Type": coupon_type,
                                "Avg Probability": prob
                            })

                    # Create DataFrame
                    df_all = pd.DataFrame(all_data)

                    # Pivot: Occupations as rows, Coupon Types as columns
                    pivot_df = df_all.pivot(index="Occupation", columns="Coupon Type", values="Avg Probability")

                    # Sort occupations by average probability across coupon types
                    pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values().index]

                    # Plot line chart
                    plt.figure(figsize=(14, max(6, len(pivot_df) * 0.3)))
                    for coupon in pivot_df.columns:
                        plt.plot(pivot_df.index, pivot_df[coupon], marker='o', label=coupon)

                    plt.xticks(rotation=90)
                    plt.ylim(0, 1)
                    plt.xlabel("Occupation")
                    plt.ylabel("Average Predicted Probability")
                    plt.title("Comparison of Predicted Probabilities by Occupation for Each Coupon Type")
                    plt.legend(title="Coupon Type", bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.grid(True)
                    plt.tight_layout()

                    # Display in Streamlit
                    st.pyplot(plt)
                    plt.clf()
                else:
                    # Plot per-coupon bar chart
                    plot_occupation_probs(tab_name, coupon_dict[tab_name])




    # ---------------- DEMOGRAPHICS ----------------
    with main_tab2:
        st.subheader("By Demographics")
        demo_tab1, demo_tab2, demo_tab3, demo_tab4, demo_tab5 = st.tabs([
            "Education", "Age Group", "Income Group", "Marital Group", "Gender"
        ])

# Education Tab
        with demo_tab1:
            st.write("Predicted probabilities by education level")

            edu_dict = {
                "Bachelors degree": [
                    ("Installation Maintenance & Repair", 0.859),
                    ("Construction & Extraction", 0.847),
                    ("Protective Service", 0.844),
                    ("Healthcare Practitioners & Technical", 0.820),
                    ("Food Preparation & Serving Related", 0.762),
                    ("Management", 0.696),
                    ("Legal", 0.656),
                    ("Education&Training&Library", 0.641),
                    ("Business & Financial", 0.627),
                    ("Office & Administrative Support", 0.549),
                    ("Student", 0.547),
                    ("Healthcare Support", 0.546),
                    ("Community & Social Services", 0.521),
                    ("Transportation & Material Moving", 0.518),
                    ("Unemployed", 0.516),
                    ("Sales & Related", 0.511),
                    ("Architecture & Engineering", 0.490),
                    ("Computer & Mathematical", 0.488),
                    ("Life Physical Social Science", 0.482),
                    ("Arts Design Entertainment Sports & Media", 0.470),
                    ("Personal Care & Service", 0.454),
                    ("Retired", 0.399),
                    ("Production Occupations", 0.298),
                ],
                "Some college - no degree": [
                    ("Life Physical Social Science", 0.861),
                    ("Construction & Extraction", 0.798),
                    ("Office & Administrative Support", 0.744),
                    ("Healthcare Support", 0.713),
                    ("Healthcare Practitioners & Technical", 0.713),
                    ("Protective Service", 0.685),
                    ("Transportation & Material Moving", 0.659),
                    ("Production Occupations", 0.629),
                    ("Computer & Mathematical", 0.613),
                    ("Student", 0.604),
                    ("Business & Financial", 0.599),
                    ("Food Preparation & Serving Related", 0.583),
                    ("Farming Fishing & Forestry", 0.576),
                    ("Sales & Related", 0.562),
                    ("Unemployed", 0.546),
                    ("Personal Care & Service", 0.538),
                    ("Management", 0.529),
                    ("Arts Design Entertainment Sports & Media", 0.480),
                    ("Retired", 0.454),
                    ("Installation Maintenance & Repair", 0.436),
                    ("Education&Training&Library", 0.322),
                    ("Building & Grounds Cleaning & Maintenance", 0.013),
                ],
                "Graduate degree (Masters or Doctorate)": [
                    ("Food Preparation & Serving Related", 0.955),
                    ("Arts Design Entertainment Sports & Media", 0.739),
                    ("Unemployed", 0.738),
                    ("Architecture & Engineering", 0.724),
                    ("Healthcare Support", 0.716),
                    ("Management", 0.684),
                    ("Legal", 0.645),
                    ("Healthcare Practitioners & Technical", 0.629),
                    ("Life Physical Social Science", 0.612),
                    ("Sales & Related", 0.575),
                    ("Business & Financial", 0.553),
                    ("Education&Training&Library", 0.544),
                    ("Retired", 0.510),
                    ("Computer & Mathematical", 0.462),
                    ("Transportation & Material Moving", 0.421),
                    ("Student", 0.396),
                    ("Community & Social Services", 0.314),
                ],
                "Associates degree": [
                    ("Healthcare Support", 0.972),
                    ("Community & Social Services", 0.954),
                    ("Student", 0.834),
                    ("Computer & Mathematical", 0.734),
                    ("Office & Administrative Support", 0.709),
                    ("Education&Training&Library", 0.670),
                    ("Management", 0.658),
                    ("Installation Maintenance & Repair", 0.653),
                    ("Healthcare Practitioners & Technical", 0.641),
                    ("Transportation & Material Moving", 0.618),
                    ("Protective Service", 0.617),
                    ("Legal", 0.596),
                    ("Food Preparation & Serving Related", 0.549),
                    ("Unemployed", 0.519),
                    ("Building & Grounds Cleaning & Maintenance", 0.500),
                    ("Retired", 0.487),
                    ("Arts Design Entertainment Sports & Media", 0.442),
                    ("Farming Fishing & Forestry", 0.413),
                    ("Sales & Related", 0.328),
                ],
                "High School Graduate": [
                    ("Installation Maintenance & Repair", 0.942),
                    ("Healthcare Support", 0.864),
                    ("Business & Financial", 0.829),
                    ("Sales & Related", 0.665),
                    ("Arts Design Entertainment Sports & Media", 0.660),
                    ("Unemployed", 0.638),
                    ("Management", 0.581),
                    ("Food Preparation & Serving Related", 0.509),
                    ("Student", 0.486),
                    ("Production Occupations", 0.443),
                    ("Transportation & Material Moving", 0.419),
                    ("Retired", 0.399),
                    ("Computer & Mathematical", 0.342),
                    ("Office & Administrative Support", 0.173),
                ],
                "Some High School": [
                    ("Computer & Mathematical", 0.767),
                    ("Unemployed", 0.667),
                ],
            }

            edu_tabs = st.tabs(list(edu_dict.keys()))
            for tab, level in zip(edu_tabs, edu_dict.keys()):
                with tab:
                    plot_occupation_probs(f"Education Level: {level}", edu_dict[level])


        # Age Group Tab
# Age Group Tab
        with demo_tab2:
            st.write("Predicted probabilities by age group")

            age_dict = {
                "Senior": [
                    ("Life Physical Social Science", 0.181),
                    ("Community & Social Services", 0.416),
                    ("Sales & Related", 0.433),
                    ("Computer & Mathematical", 0.448),
                    ("Retired", 0.448),
                    ("Education&Training&Library", 0.479),
                    ("Unemployed", 0.503),
                    ("Arts Design Entertainment Sports & Media", 0.513),
                    ("Construction & Extraction", 0.514),
                    ("Personal Care & Service", 0.531),
                    ("Transportation & Material Moving", 0.559),
                    ("Management", 0.577),
                    ("Business & Financial", 0.580),
                    ("Food Preparation & Serving Related", 0.582),
                    ("Legal", 0.607),
                    ("Installation Maintenance & Repair", 0.610),
                    ("Healthcare Practitioners & Technical", 0.793),
                    ("Office & Administrative Support", 0.795),
                    ("Protective Service", 0.883),
                    ("Healthcare Support", 0.917),
                    ("Production Occupations", 0.998),
                ],
                "Adult": [
                    ("Building & Grounds Cleaning & Maintenance", 0.208),
                    ("Personal Care & Service", 0.346),
                    ("Production Occupations", 0.379),
                    ("Retired", 0.402),
                    ("Arts Design Entertainment Sports & Media", 0.438),
                    ("Community & Social Services", 0.470),
                    ("Student", 0.486),
                    ("Computer & Mathematical", 0.500),
                    ("Sales & Related", 0.568),
                    ("Farming Fishing & Forestry", 0.576),
                    ("Unemployed", 0.592),
                    ("Office & Administrative Support", 0.598),
                    ("Education&Training&Library", 0.621),
                    ("Legal", 0.638),
                    ("Life Physical Social Science", 0.642),
                    ("Food Preparation & Serving Related", 0.642),
                    ("Installation Maintenance & Repair", 0.653),
                    ("Transportation & Material Moving", 0.658),
                    ("Management", 0.667),
                    ("Healthcare Practitioners & Technical", 0.681),
                    ("Architecture & Engineering", 0.720),
                    ("Business & Financial", 0.739),
                    ("Protective Service", 0.754),
                    ("Healthcare Support", 0.772),
                    ("Construction & Extraction", 0.861),
                ],
                "Young": [
                    ("Architecture & Engineering", 0.198),
                    ("Education&Training&Library", 0.407),
                    ("Life Physical Social Science", 0.511),
                    ("Food Preparation & Serving Related", 0.521),
                    ("Office & Administrative Support", 0.521),
                    ("Healthcare Support", 0.524),
                    ("Community & Social Services", 0.526),
                    ("Business & Financial", 0.531),
                    ("Unemployed", 0.549),
                    ("Arts Design Entertainment Sports & Media", 0.575),
                    ("Sales & Related", 0.579),
                    ("Student", 0.593),
                    ("Personal Care & Service", 0.641),
                    ("Management", 0.693),
                    ("Computer & Mathematical", 0.723),
                    ("Protective Service", 0.735),
                    ("Legal", 0.737),
                ],
                "MiddleAged": [
                    ("Personal Care & Service", 0.382),
                    ("Farming Fishing & Forestry", 0.413),
                    ("Community & Social Services", 0.446),
                    ("Sales & Related", 0.451),
                    ("Student", 0.461),
                    ("Protective Service", 0.513),
                    ("Arts Design Entertainment Sports & Media", 0.527),
                    ("Retired", 0.539),
                    ("Business & Financial", 0.540),
                    ("Unemployed", 0.541),
                    ("Healthcare Support", 0.546),
                    ("Transportation & Material Moving", 0.565),
                    ("Production Occupations", 0.567),
                    ("Education&Training&Library", 0.571),
                    ("Legal", 0.615),
                    ("Management", 0.661),
                    ("Architecture & Engineering", 0.688),
                    ("Computer & Mathematical", 0.753),
                    ("Office & Administrative Support", 0.785),
                    ("Construction & Extraction", 0.847),
                    ("Installation Maintenance & Repair", 0.942),
                    ("Food Preparation & Serving Related", 0.955),
                ],
            }

            age_tabs = st.tabs(list(age_dict.keys()))
            for tab, group in zip(age_tabs, age_dict.keys()):
                with tab:
                    plot_occupation_probs(f"Age Group: {group}", age_dict[group])


# Income Group Tab
        with demo_tab3:
            st.write("Predicted probabilities by income group")

            # Data dictionary for each income group
            income_dict = {
                "Low": [
                    ("Life Physical Social Science", 0.861),
                    ("Construction & Extraction", 0.800),
                    ("Healthcare Support", 0.756),
                    ("Legal", 0.752),
                    ("Computer & Mathematical", 0.698),
                    ("Management", 0.676),
                    ("Protective Service", 0.661),
                    ("Installation Maintenance & Repair", 0.660),
                    ("Business & Financial", 0.658),
                    ("Healthcare Practitioners & Technical", 0.619),
                    ("Transportation & Material Moving", 0.616),
                    ("Food Preparation & Serving Related", 0.599),
                    ("Education&Training&Library", 0.598),
                    ("Community & Social Services", 0.585),
                    ("Unemployed", 0.580),
                    ("Office & Administrative Support", 0.572),
                    ("Retired", 0.555),
                    ("Student", 0.549),
                    ("Sales & Related", 0.548),
                    ("Arts Design Entertainment Sports & Media", 0.546),
                    ("Personal Care & Service", 0.525),
                    ("Production Occupations", 0.511),
                    ("Farming Fishing & Forestry", 0.413),
                    ("Building & Grounds Cleaning & Maintenance", 0.013),
                ],
                "Medium": [
                    ("Protective Service", 0.883),
                    ("Construction & Extraction", 0.875),
                    ("Healthcare Practitioners & Technical", 0.871),
                    ("Food Preparation & Serving Related", 0.828),
                    ("Office & Administrative Support", 0.777),
                    ("Healthcare Support", 0.726),
                    ("Legal", 0.701),
                    ("Transportation & Material Moving", 0.665),
                    ("Installation Maintenance & Repair", 0.653),
                    ("Student", 0.628),
                    ("Management", 0.621),
                    ("Business & Financial", 0.614),
                    ("Education&Training&Library", 0.587),
                    ("Farming Fishing & Forestry", 0.576),
                    ("Unemployed", 0.563),
                    ("Life Physical Social Science", 0.541),
                    ("Sales & Related", 0.523),
                    ("Personal Care & Service", 0.504),
                    ("Building & Grounds Cleaning & Maintenance", 0.500),
                    ("Computer & Mathematical", 0.498),
                    ("Arts Design Entertainment Sports & Media", 0.483),
                    ("Retired", 0.481),
                    ("Community & Social Services", 0.481),
                    ("Production Occupations", 0.438),
                    ("Architecture & Engineering", 0.414),
                ],
                "High": [
                    ("Protective Service", 0.754),
                    ("Healthcare Practitioners & Technical", 0.686),
                    ("Management", 0.671),
                    ("Construction & Extraction", 0.669),
                    ("Business & Financial", 0.631),
                    ("Student", 0.617),
                    ("Architecture & Engineering", 0.613),
                    ("Legal", 0.605),
                    ("Office & Administrative Support", 0.589),
                    ("Sales & Related", 0.547),
                    ("Healthcare Support", 0.546),
                    ("Unemployed", 0.539),
                    ("Computer & Mathematical", 0.524),
                    ("Education&Training&Library", 0.478),
                    ("Life Physical Social Science", 0.456),
                    ("Personal Care & Service", 0.455),
                    ("Transportation & Material Moving", 0.452),
                    ("Food Preparation & Serving Related", 0.450),
                    ("Arts Design Entertainment Sports & Media", 0.444),
                    ("Community & Social Services", 0.322),
                    ("Retired", 0.232),
                ]
            }

            # Create sub-tabs for each income group and plot inside
            income_tabs = st.tabs(list(income_dict.keys()))
            for tab, group in zip(income_tabs, income_dict.keys()):
                with tab:
                    plot_occupation_probs(f"Income Group: {group}", income_dict[group])


        # Marital Group Tab
# Marital Group Tab
        with demo_tab4:
            st.write("Predicted probabilities by marital group")

            marital_dict = {
                "Partnered": [
                    ("Building & Grounds Cleaning & Maintenance", 0.208),
                    ("Community & Social Services", 0.410),
                    ("Retired", 0.439),
                    ("Personal Care & Service", 0.474),
                    ("Unemployed", 0.494),
                    ("Farming Fishing & Forestry", 0.495),
                    ("Student", 0.502),
                    ("Arts Design Entertainment Sports & Media", 0.519),
                    ("Installation Maintenance & Repair", 0.524),
                    ("Computer & Mathematical", 0.529),
                    ("Transportation & Material Moving", 0.555),
                    ("Life Physical Social Science", 0.570),
                    ("Sales & Related", 0.573),
                    ("Education&Training&Library", 0.576),
                    ("Office & Administrative Support", 0.608),
                    ("Legal", 0.638),
                    ("Business & Financial", 0.641),
                    ("Architecture & Engineering", 0.644),
                    ("Healthcare Practitioners & Technical", 0.644),
                    ("Management", 0.663),
                    ("Healthcare Support", 0.669),
                    ("Food Preparation & Serving Related", 0.670),
                    ("Protective Service", 0.761),
                    ("Production Occupations", 0.773),
                    ("Construction & Extraction", 0.851),
                ],
                "Single": [
                    ("Architecture & Engineering", 0.197),
                    ("Production Occupations", 0.447),
                    ("Arts Design Entertainment Sports & Media", 0.465),
                    ("Retired", 0.472),
                    ("Education&Training&Library", 0.509),
                    ("Sales & Related", 0.511),
                    ("Personal Care & Service", 0.525),
                    ("Food Preparation & Serving Related", 0.544),
                    ("Life Physical Social Science", 0.567),
                    ("Computer & Mathematical", 0.601),
                    ("Business & Financial", 0.617),
                    ("Student", 0.619),
                    ("Protective Service", 0.623),
                    ("Management", 0.642),
                    ("Unemployed", 0.657),
                    ("Legal", 0.659),
                    ("Community & Social Services", 0.660),
                    ("Office & Administrative Support", 0.669),
                    ("Transportation & Material Moving", 0.692),
                    ("Installation Maintenance & Repair", 0.711),
                    ("Construction & Extraction", 0.720),
                    ("Healthcare Support", 0.729),
                    ("Healthcare Practitioners & Technical", 0.816),
                ],
            }

            marital_tabs = st.tabs(list(marital_dict.keys()))
            for tab, group in zip(marital_tabs, marital_dict.keys()):
                with tab:
                    plot_occupation_probs(f"Marital Group: {group}", marital_dict[group])



# Gender Tab
        with demo_tab5:
            st.write("Predicted probabilities by gender")

            gender_dict = {
                "Female": [
                    ("Production Occupations", 0.298),
                    ("Personal Care & Service", 0.455),
                    ("Retired", 0.456),
                    ("Arts Design Entertainment Sports & Media", 0.473),
                    ("Transportation & Material Moving", 0.475),
                    ("Community & Social Services", 0.483),
                    ("Sales & Related", 0.488),
                    ("Life Physical Social Science", 0.541),
                    ("Student", 0.542),
                    ("Education&Training&Library", 0.542),
                    ("Unemployed", 0.546),
                    ("Computer & Mathematical", 0.546),
                    ("Business & Financial", 0.579),
                    ("Management", 0.615),
                    ("Legal", 0.625),
                    ("Office & Administrative Support", 0.634),
                    ("Food Preparation & Serving Related", 0.667),
                    ("Protective Service", 0.672),
                    ("Healthcare Practitioners & Technical", 0.725),
                    ("Healthcare Support", 0.733),
                    ("Installation Maintenance & Repair", 0.859),
                    ("Construction & Extraction", 0.895),
                    ("Architecture & Engineering", 0.921),
                ],
                "Male": [
                    ("Building & Grounds Cleaning & Maintenance", 0.208),
                    ("Food Preparation & Serving Related", 0.433),
                    ("Retired", 0.446),
                    ("Community & Social Services", 0.456),
                    ("Farming Fishing & Forestry", 0.495),
                    ("Architecture & Engineering", 0.528),
                    ("Personal Care & Service", 0.542),
                    ("Computer & Mathematical", 0.549),
                    ("Production Occupations", 0.552),
                    ("Arts Design Entertainment Sports & Media", 0.572),
                    ("Sales & Related", 0.578),
                    ("Installation Maintenance & Repair", 0.580),
                    ("Education&Training&Library", 0.591),
                    ("Student", 0.600),
                    ("Unemployed", 0.604),
                    ("Healthcare Support", 0.605),
                    ("Healthcare Practitioners & Technical", 0.625),
                    ("Life Physical Social Science", 0.644),
                    ("Transportation & Material Moving", 0.653),
                    ("Office & Administrative Support", 0.680),
                    ("Business & Financial", 0.680),
                    ("Management", 0.682),
                    ("Protective Service", 0.730),
                    ("Legal", 0.737),
                    ("Construction & Extraction", 0.779),
                ],
            }

            gender_tabs = st.tabs(list(gender_dict.keys()))
            for tab, gender in zip(gender_tabs, gender_dict.keys()):
                with tab:
                    plot_occupation_probs(f"Gender: {gender}", gender_dict[gender])


    # ---------------- CONTEXTUAL ----------------
    with main_tab3:
        st.subheader("By Contextual Features")
        context_tab1, context_tab2 = st.tabs(["Weather", "Time of Day"])

# Weather Tab
        with context_tab1:
            st.write("Predicted probabilities by weather condition")

            weather_dict = {
                "Rainy": [
                    ("Building & Grounds Cleaning & Maintenance", 0.014),
                    ("Protective Service", 0.023),
                    ("Healthcare Practitioners & Technical", 0.269),
                    ("Arts Design Entertainment Sports & Media", 0.277),
                    ("Computer & Mathematical", 0.315),
                    ("Community & Social Services", 0.377),
                    ("Legal", 0.380),
                    ("Retired", 0.383),
                    ("Unemployed", 0.419),
                    ("Healthcare Support", 0.465),
                    ("Student", 0.466),
                    ("Sales & Related", 0.469),
                    ("Farming Fishing & Forestry", 0.494),
                    ("Education&Training&Library", 0.505),
                    ("Personal Care & Service", 0.542),
                    ("Transportation & Material Moving", 0.548),
                    ("Business & Financial", 0.622),
                    ("Office & Administrative Support", 0.647),
                    ("Installation Maintenance & Repair", 0.672),
                    ("Production Occupations", 0.678),
                    ("Food Preparation & Serving Related", 0.698),
                    ("Life Physical Social Science", 0.702),
                    ("Management", 0.756),
                    ("Construction & Extraction", 0.880),
                    ("Architecture & Engineering", 0.917),
                ],
                "Sunny": [
                    ("Building & Grounds Cleaning & Maintenance", 0.013),
                    ("Production Occupations", 0.426),
                    ("Retired", 0.483),
                    ("Personal Care & Service", 0.497),
                    ("Community & Social Services", 0.497),
                    ("Architecture & Engineering", 0.519),
                    ("Sales & Related", 0.535),
                    ("Arts Design Entertainment Sports & Media", 0.546),
                    ("Life Physical Social Science", 0.560),
                    ("Education&Training&Library", 0.578),
                    ("Computer & Mathematical", 0.588),
                    ("Farming Fishing & Forestry", 0.591),
                    ("Student", 0.600),
                    ("Unemployed", 0.611),
                    ("Transportation & Material Moving", 0.636),
                    ("Food Preparation & Serving Related", 0.638),
                    ("Legal", 0.658),
                    ("Office & Administrative Support", 0.660),
                    ("Management", 0.661),
                    ("Business & Financial", 0.683),
                    ("Installation Maintenance & Repair", 0.700),
                    ("Healthcare Support", 0.711),
                    ("Healthcare Practitioners & Technical", 0.720),
                    ("Protective Service", 0.736),
                    ("Construction & Extraction", 0.786),
                ],
                "Snowy": [
                    ("Retired", 0.215),
                    ("Arts Design Entertainment Sports & Media", 0.271),
                    ("Business & Financial", 0.327),
                    ("Community & Social Services", 0.350),
                    ("Food Preparation & Serving Related", 0.350),
                    ("Computer & Mathematical", 0.382),
                    ("Installation Maintenance & Repair", 0.383),
                    ("Farming Fishing & Forestry", 0.398),
                    ("Unemployed", 0.414),
                    ("Personal Care & Service", 0.422),
                    ("Education&Training&Library", 0.480),
                    ("Building & Grounds Cleaning & Maintenance", 0.500),
                    ("Student", 0.510),
                    ("Transportation & Material Moving", 0.558),
                    ("Office & Administrative Support", 0.561),
                    ("Management", 0.567),
                    ("Sales & Related", 0.609),
                    ("Architecture & Engineering", 0.666),
                    ("Production Occupations", 0.672),
                    ("Legal", 0.692),
                    ("Construction & Extraction", 0.736),
                    ("Healthcare Practitioners & Technical", 0.810),
                    ("Healthcare Support", 0.903),
                ]
            }

            weather_tabs = st.tabs(list(weather_dict.keys()))
            for tab, weather in zip(weather_tabs, weather_dict.keys()):
                with tab:
                    plot_occupation_probs(f"Weather: {weather}", weather_dict[weather])


        with context_tab2:
            st.write("Predicted probabilities by time of day")

            time_of_day_dict = {
                "Morning": [
                    ("Protective Service", 0.374),
                    ("Retired", 0.396),
                    ("Community & Social Services", 0.434),
                    ("Arts Design Entertainment Sports & Media", 0.476),
                    ("Personal Care & Service", 0.492),
                    ("Education&Training&Library", 0.502),
                    ("Computer & Mathematical", 0.511),
                    ("Unemployed", 0.515),
                    ("Architecture & Engineering", 0.538),
                    ("Sales & Related", 0.540),
                    ("Student", 0.544),
                    ("Life Physical Social Science", 0.556),
                    ("Food Preparation & Serving Related", 0.570),
                    ("Transportation & Material Moving", 0.570),
                    ("Production Occupations", 0.570),
                    ("Legal", 0.571),
                    ("Office & Administrative Support", 0.615),
                    ("Business & Financial", 0.625),
                    ("Management", 0.642),
                    ("Installation Maintenance & Repair", 0.656),
                    ("Construction & Extraction", 0.673),
                    ("Healthcare Support", 0.700),
                    ("Healthcare Practitioners & Technical", 0.704),
                    ("Farming Fishing & Forestry", 0.724),
                ],
                "Afternoon": [
                    ("Building & Grounds Cleaning & Maintenance", 0.022),
                    ("Food Preparation & Serving Related", 0.525),
                    ("Sales & Related", 0.594),
                    ("Architecture & Engineering", 0.596),
                    ("Education&Training&Library", 0.608),
                    ("Transportation & Material Moving", 0.615),
                    ("Retired", 0.621),
                    ("Computer & Mathematical", 0.644),
                    ("Business & Financial", 0.660),
                    ("Arts Design Entertainment Sports & Media", 0.670),
                    ("Community & Social Services", 0.677),
                    ("Student", 0.679),
                    ("Management", 0.727),
                    ("Protective Service", 0.749),
                    ("Healthcare Practitioners & Technical", 0.749),
                    ("Personal Care & Service", 0.753),
                    ("Life Physical Social Science", 0.767),
                    ("Office & Administrative Support", 0.790),
                    ("Unemployed", 0.862),
                    ("Legal", 0.891),
                    ("Construction & Extraction", 0.962),
                    ("Healthcare Support", 0.970),
                    ("Installation Maintenance & Repair", 0.999),
                ],
                "Evening": [
                    ("Building & Grounds Cleaning & Maintenance", 0.255),
                    ("Production Occupations", 0.382),
                    ("Life Physical Social Science", 0.415),
                    ("Retired", 0.423),
                    ("Arts Design Entertainment Sports & Media", 0.428),
                    ("Personal Care & Service", 0.436),
                    ("Community & Social Services", 0.447),
                    ("Farming Fishing & Forestry", 0.474),
                    ("Sales & Related", 0.519),
                    ("Unemployed", 0.534),
                    ("Computer & Mathematical", 0.549),
                    ("Architecture & Engineering", 0.556),
                    ("Student", 0.582),
                    ("Installation Maintenance & Repair", 0.587),
                    ("Legal", 0.596),
                    ("Education&Training&Library", 0.618),
                    ("Office & Administrative Support", 0.621),
                    ("Business & Financial", 0.626),
                    ("Management", 0.636),
                    ("Healthcare Practitioners & Technical", 0.646),
                    ("Healthcare Support", 0.649),
                    ("Transportation & Material Moving", 0.656),
                    ("Food Preparation & Serving Related", 0.673),
                    ("Protective Service", 0.895),
                    ("Construction & Extraction", 0.907),
                ]
            }

            time_tabs = st.tabs(list(time_of_day_dict.keys()))
            for tab, time in zip(time_tabs, time_of_day_dict.keys()):
                with tab:
                    plot_occupation_probs(f"Time of Day: {time}", time_of_day_dict[time])


