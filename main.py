uploaded_file = st.file_uploader(
    "Upload your pet adoption CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")
    st.dataframe(df.head())
else:
    st.info("Please upload a pet adoption CSV file to begin.")
