# 📊 StatScope AutoML: 

StatScope is a fast, no-code AutoML tool built with Streamlit that lets you:
- Upload your dataset (CSV)
- Automatically detect date/time columns
- Automatically clean, encode, and preprocess your data
- Choose the target column and select the features for training
- Automatically build and tune a Random Forest Regressor
- Evaluate and visualize model performance
- Download the trained model and predictions

⸻

🚀 How to run the app locally

1️⃣ Clone the repository

git clone https://github.com/rishiraj333/StatScope_AutoML.git
cd StatScope_AutoML

2️⃣ Set up the environment

If using conda:

conda env create -f environment.yml
conda activate statscope_env

Or using pip:

python -m venv statscope_env
source statscope_env/bin/activate   # On Windows: statscope_env\Scripts\activate
pip install -r requirements.txt

3️⃣ Run the Streamlit app

streamlit run app.py


⸻

📂 How to use the app

1️⃣ Upload your dataset
- Click “Browse files” to upload a CSV file.
- Your dataset will load, and you’ll see a preview.

2️⃣ Select target and features
- Choose your target variable (what you want to predict).
- Optionally, choose specific features (or leave as default = all).

3️⃣ Run AutoML
- Click “🚀 Run AutoML”. The app will:
- Clean data (handle missing values, duplicates)
- Encode categorical variables
- Scale features
- Auto-tune the Random Forest model
- Show performance metrics + plots

4️⃣ Download results
- Download the trained model (.pkl)
- Download the predictions (.csv)

⸻

📈 Outputs
- R² score: How well the model explains variance in the data
- MSE: How large the prediction errors are
- Actual vs. Predicted plot (scatter + line)

⸻

📝 Example dataset format

Date    Feature1    Feature2    Target
2023-01-01    10    0.5    100
2023-01-02    12    0.7    110

✅ Date/time columns will be automatically detected and processed!

⸻

💡 Notes
- The app currently supports regression tasks only.
- This version is still under testing phase, so feel free to try and test various datasets to check for bugs.

⸻

📌 Requirements
- Python >= 3.8
- Streamlit
- pandas
- scikit-learn
- seaborn
- matplotlib
- numpy
- pickle

(See environment.yml or requirements.txt for full list)

⸻

🙌 Contributing

Feel free to fork this repo and submit pull requests for improvements!

⸻

📬 Contact

For any questions or suggestions, please reach out via GitHub issues or email rishirajtripathi333@gmail.com !
