import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.data_loading import load_sample_data, load_from_csv, save_processed_data
from src.preprocessing import TextPreprocessor
from src.evaluation import evaluate_model, plot_confusion_matrix, classification_report_df, CONFUSION_MATRIX_PATH
from src.inference import SentimentAnalyzer
from src.ticket_classification import TicketClassifier

# Page config
st.set_page_config(
    page_title="📊 Sentiment Analysis App",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 0.5rem;
    }
    .sentiment-positive {
        color: #2ca02c;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #d62728;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Sentiment Analysis & Ticket Classification")
st.markdown("---")

# Sidebar
st.sidebar.title("Menu")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["🏠 Home", "📝 Single Analysis", "📊 Batch Analysis", "📈 Model Evaluation", "📁 Data Management", "🎫 Ticket Classification", "🏋️ Train Model"]
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()

# Load sample data button
if st.sidebar.button("📥 Load Sample Data"):
    df = load_sample_data()
    st.session_state.sample_data = df
    st.success(f"Loaded {len(df)} sample samples")

# ---- HOME PAGE ----
if app_mode == "🏠 Home":
    st.header("Welcome to Sentiment Analysis")
    st.markdown("""
    This application performs sentiment analysis on text data using 
    fine-tuned BERT models. You can:
    
    - Analyze sentiment of single text entries
    - Process batch analysis from CSV files
    - Evaluate model performance
    - Manage and preprocess data
    
    **Features:**
    - Real-time sentiment prediction (Positive/Negative)
    - Confidence scores
    - Text preprocessing pipeline
    - Batch processing support
    - Evaluation metrics and confusion matrix
    """)
    
    # Show sample data if available
    if st.session_state.get('sample_data') is not None:
        st.subheader("Sample Data Preview")
        st.dataframe(st.session_state.sample_data.head(10))

# ---- SINGLE ANALYSIS ----
elif app_mode == "📝 Single Analysis":
    st.header("Single Text Analysis")
    
    # Text input
    user_text = st.text_area(
        "Enter text for sentiment analysis:",
        height=150,
        placeholder="Type or paste text here..."
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_btn = st.button("Analyze Sentiment", type="primary")
    with col2:
        clear_btn = st.button("Clear")
    
    if analyze_btn and user_text:
        if st.session_state.analyzer is None:
            with st.spinner("Loading model..."):
                st.session_state.analyzer = SentimentAnalyzer()
        
        with st.spinner("Analyzing..."):
            result = st.session_state.analyzer.analyze(user_text)
        
        # Display result
        st.markdown("---")
        st.subheader("Result")
        
        sentiment = result['sentiment']
        confidence = result['confidence']
        
        if sentiment == "Positive":
            st.markdown(f'<p class="sentiment-positive">🟢 {sentiment} (Confidence: {confidence:.2%})</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="sentiment-negative">🔴 {sentiment} (Confidence: {confidence:.2%})</p>', 
                       unsafe_allow_html=True)
        
        # Probability bars
        st.markdown("#### Probability Distribution")
        prob_df = pd.DataFrame({
            'Sentiment': ['Negative', 'Positive'],
            'Probability': [result['probabilities']['negative'], result['probabilities']['positive']]
        })
        
        fig = px.bar(prob_df, x='Sentiment', y='Probability', 
                     color='Sentiment', color_discrete_map={'Negative': '#d62728', 'Positive': '#2ca02c'},
                     text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show cleaned text
        with st.expander("Show preprocessing details"):
            st.write(f"**Original text:** {user_text}")
            st.write(f"**Cleaned text:** {result['cleaned_text']}")
    
    if clear_btn:
        st.rerun()

# ---- BATCH ANALYSIS ----
elif app_mode == "📊 Batch Analysis":
    st.header("Batch Text Analysis")
    
    st.markdown("Upload a CSV file with text data (column: 'text') or use sample data.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} rows.")
            st.dataframe(df.head())
            
            if st.button("Run Batch Analysis"):
                if st.session_state.analyzer is None:
                    with st.spinner("Loading model..."):
                        st.session_state.analyzer = SentimentAnalyzer()
                
                with st.spinner("Processing batch..."):
                    texts = df['text'].astype(str).tolist()
                    results = st.session_state.analyzer.batch_analyze(texts)
                
                # Add results to dataframe
                df['predicted_sentiment'] = [r['sentiment'] for r in results]
                df['confidence'] = [r['confidence'] for r in results]
                
                st.subheader("Results")
                st.dataframe(df)
                
                # Summary stats
                pos_count = sum(1 for r in results if r['sentiment'] == 'Positive')
                neg_count = sum(1 for r in results if r['sentiment'] == 'Negative')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyzed", len(results))
                with col2:
                    st.metric("Positive", pos_count, f"{pos_count/len(results):.1%}")
                with col3:
                    st.metric("Negative", neg_count, f"{neg_count/len(results):.1%}")
                
                # Download button
                csv = df.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    if st.session_state.get('sample_data') is not None and st.button("Use Sample Data"):
        df = st.session_state.sample_data
        st.session_state.analyzer = SentimentAnalyzer()
        results = st.session_state.analyzer.batch_analyze(df['text'].astype(str).tolist())
        
        df['predicted_sentiment'] = [r['sentiment'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        st.dataframe(df)

# ---- MODEL EVALUATION ----
elif app_mode == "📈 Model Evaluation":
    st.header("Model Evaluation")
    
    st.markdown("""
    Evaluate the trained model on test data and view performance metrics.
    """)
    
    # Load sample data for evaluation
    if st.button("Evaluate with Sample Data"):
        if st.session_state.analyzer is None:
            st.session_state.analyzer = SentimentAnalyzer()
        
        df = load_sample_data()
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Evaluate
        with st.spinner("Evaluating model..."):
            # Get predictions
            val_texts = val_df['text'].astype(str).tolist()
            results = st.session_state.analyzer.batch_analyze(val_texts)
            
            # True labels
            y_true = val_df['label'].tolist()
            y_pred = [0 if r['sentiment'] == 'Negative' else 1 for r in results]
            
            # Evaluation metrics
            metrics = evaluate_model(y_true, y_pred, ['Negative', 'Positive'])
            
            # Display overall accuracy
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['overall']['accuracy']:.2%}")
            
            # Per-class metrics
            st.subheader("Per-Class Metrics")
            report_df = classification_report_df(metrics['report'])
            st.dataframe(report_df.round(4))
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = plot_confusion_matrix(y_true, y_pred)
            st.image(str(CONFUSION_MATRIX_PATH))
            
            # Classification report text
            st.subheader("Classification Report")
            st.text(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# ---- DATA MANAGEMENT ----
elif app_mode == "📁 Data Management":
    st.header("Data Management")
    
    st.markdown("""
    Manage and preprocess your dataset for sentiment analysis.
    """)
    
    # Upload custom dataset
    uploaded_data = st.file_uploader("Upload CSV for preprocessing", type="csv", key="management")
    
    if uploaded_data is not None:
        try:
            df = pd.read_csv(uploaded_data)
            st.dataframe(df.head(10))
            
            # Preprocessing options
            st.subheader("Preprocessing Options")
            clean_check = st.checkbox("Clean text (lowercase, remove URLs)", value=True)
            stopwords_check = st.checkbox("Remove stopwords", value=True)
            stemming_check = st.checkbox("Apply stemming", value=False)
            lemmatization_check = st.checkbox("Apply lemmatization", value=False)
            
            preprocessor = TextPreprocessor(language='english')
            
            if st.button("Apply Preprocessing"):
                with st.spinner("Processing..."):
                    df['cleaned_text'] = df['text'].apply(
                        lambda x: preprocessor.preprocess(
                            str(x),
                            clean=clean_check,
                            remove_stopwords_flag=stopwords_check,
                            stem=stemming_check,
                            lemmatize=lemmatization_check
                        )
                    )
                
                st.dataframe(df[['text', 'cleaned_text']].head())
                
                # Save button
                if st.button("Save Preprocessed Data"):
                    save_processed_data(df, "data/preprocessed_data.csv")
                    st.success("Data saved to data/preprocessed_data.csv")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ---- TRAIN MODEL ----
elif app_mode == "🏋️ Train Model":
    st.header("Model Training")
    
    st.markdown("""
    Train a BERT-based sentiment analysis model on your data.
    """)
    
    # Initialize session state
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Load sample data button
    if st.button("Load Sample Data for Training"):
        from src.data_loading import load_sample_data
        import torch
        from transformers import AutoTokenizer
        from src.model_training import BERTweetClassifier, Trainer
        
        df = load_sample_data()
        st.session_state.sample_df = df
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Initialize model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BERTweetClassifier(num_classes=2).to(device)
        
        st.session_state.trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            val_df=val_df,
            epochs=3,
            learning_rate=2e-5,
            batch_size=16,
            device=device
        )
        st.session_state.model_loaded = True
        st.success("Trainer initialized with sample data!")
    
    # Train button
    if st.session_state.get('model_loaded') and st.session_state.trainer is not None:
        if st.button("Start Training", type="primary"):
            import time
            with st.spinner("Training in progress..."):
                start_time = time.time()
                history = st.session_state.trainer.train(show_progress=False)
                elapsed = time.time() - start_time
            
            # Evaluate
            metrics = st.session_state.trainer.evaluate(st.session_state.sample_df)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Accuracy", f"{metrics['accuracy']:.2%}")
            with col2:
                st.metric("Training Time", f"{elapsed:.1f}s")
            with col3:
                st.metric("Epochs", len(history['train_loss']))
            
            # Plot loss
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['train_loss'], name="Train Loss", mode='lines'))
            fig.add_trace(go.Scatter(y=history['val_loss'], name="Val Loss", mode='lines'))
            fig.update_layout(title="Training & Validation Loss", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=history['val_accuracy'], name="Val Accuracy", mode='lines', line=dict(color='green')))
            fig2.update_layout(title="Validation Accuracy", yaxis_title="Accuracy")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Save model
            st.session_state.trainer.save_model()
            st.success("✅ Model trained and saved successfully!")
    
    # Evaluation metrics info
    if st.session_state.get('model_loaded') and st.session_state.trainer is not None:
        st.markdown("---")
        st.subheader("Last Training Stats")
        # Show placeholder if no training yet
        st.info("Train a model to see statistics here")

# ---- TICKET CLASSIFICATION ----
elif app_mode == "🎫 Ticket Classification":
    st.header("Ticket Classification")
    
    st.markdown("""
    Classify support tickets into categories for routing and prioritization.
    """)
    
    # Initialize classifier if not already (prefer trained ML model)
    if 'ticket_classifier' not in st.session_state:
        classifier = TicketClassifier()
        if classifier.load_ml():
            st.sidebar.success("ML ticket model loaded")
        st.session_state.ticket_classifier = classifier
    
    classifier = st.session_state.ticket_classifier
    
    # Text input
    ticket_text = st.text_area(
        "Enter support ticket text:",
        height=150,
        placeholder="Type or paste the support ticket text here..."
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        classify_btn = st.button("Classify Ticket", type="primary")
    with col2:
        clear_ticket_btn = st.button("Clear")
    
    if classify_btn and ticket_text:
        with st.spinner("Classifying ticket..."):
            result = classifier.classify(ticket_text)
        
        # Display result
        st.markdown("---")
        st.subheader("Classification Result")
        
        category = result['category']
        priority = result['priority']
        confidence = result['confidence']
        
        # Category color coding
        category_colors = {
            "Technical Issues": "#d62728",  # red
            "Billing & Payments": "#ff7f0e",  # orange
            "Account Access": "#9467bd",  # purple
            "Feature Requests": "#2ca02c",  # green
            "General Inquiries": "#8c564b",  # brown
            "Urgent/Support Priority": "#e377c2",  # pink
        }
        
        color = category_colors.get(category, "#1f77b4")
        
        st.markdown(f'''
        <div style="
            padding: 1rem;
            background-color: {color}05;
            border-left: 4px solid {color};
            border-radius: 4px;
            margin: 1rem 0;
        ">
            <h3 style="color: {color}; margin: 0;">{category}</h3>
            <p style="margin: 0.5rem 0;">Priority: <strong>{priority.upper()}</strong></p>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">Confidence: {confidence:.2%}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Probability/keyword breakdown
        with st.expander("Show classification details"):
            st.write(f"**Original text:** {ticket_text}")
            
            # Get keywords that influenced the classification
            cleaned = classifier.preprocessor.clean_ticket_text(ticket_text)
            keywords = classifier.preprocessor.extract_keywords(cleaned)
            
            st.write("**Keywords detected:**")
            if keywords:
                kw_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
                st.dataframe(kw_df)
            else:
                st.write("No significant keywords found.")
            
            # Show all category scores
            st.write("**Category assessment:**")
            # Simple re-assessment
            category_keywords = classifier.category_keywords
            for cat, kws in category_keywords.items():
                matches = [kw for kw in kws if kw in {w[0] for w in keywords}]
                if matches:
                    st.write(f"  - {cat}: {len(matches)} matching keywords")
    
    if clear_ticket_btn:
        st.rerun()
    
    # Sample tickets demo
    st.markdown("---")
    st.subheader("Sample Tickets")
    
    sample_tickets = [
        "Payment charged twice, need refund immediately!",
        "Forgot password, can't access account",
        "Love the app, want dark mode feature",
        "App crashes every time I try to upload",
        "Hello, question about subscription plan",
        "System error at checkout, help needed!"
    ]
    
    cols = st.columns(3)
    for i, ticket in enumerate(sample_tickets):
        col = cols[i % 3]
        if col.button(ticket[:40] + ("..." if len(ticket) > 40 else ""), key=f"sample_{i}"):
            with st.spinner("Classifying..."):
                result = classifier.classify(ticket)
            st.session_state['last_result'] = result
            st.rerun()
    
    if st.session_state.get('last_result'):
        result = st.session_state['last_result']
        st.info(f"Last classified: {result['category']} (Priority: {result['priority']}, Conf: {result['confidence']:.0%})")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <small>Sentiment Analysis & Ticket Classification Project | Built with Streamlit & Transformers</small>
</div>
""", unsafe_allow_html=True)