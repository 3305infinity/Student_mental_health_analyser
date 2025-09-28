
# Student Mental Health Analyzer Pro

An advanced machine learning system for mental health risk assessment in academic environments.

## Overview

This application employs ensemble machine learning methodologies to analyze student mental health indicators and provide data-driven risk assessments. The system integrates multiple algorithmic approaches with natural language processing capabilities for comprehensive mental health evaluation.

**IMPORTANT MEDICAL DISCLAIMER**: This software is designed for research and educational purposes only. It does not provide medical diagnosis or treatment recommendations. Users experiencing mental health crises should consult qualified healthcare professionals immediately.

## Technical Specifications

### Core Architecture
- **Machine Learning Pipeline**: 8 comparative algorithms with automated model selection
- **Data Processing**: Scikit-learn preprocessing with categorical encoding and feature scaling
- **AI Integration**: Google Gemini API for natural language recommendations
- **Interface**: Streamlit-based web application with interactive visualizations

### Algorithmic Framework
| Algorithm | Implementation | Primary Use Case |
|-----------|---------------|------------------|
| Random Forest | Ensemble bagging | Feature importance analysis |
| Gradient Boosting | Sequential boosting | Complex pattern recognition |
| Support Vector Machine | Kernel-based classification | High-dimensional separation |
| Logistic Regression | Linear probabilistic model | Baseline comparison |
| Decision Tree | CART algorithm | Rule extraction |
| K-Nearest Neighbors | Distance-based classification | Local pattern matching |
| Naive Bayes | Probabilistic classifier | Categorical feature handling |
| AdaBoost | Adaptive boosting | Weak learner enhancement |

## System Requirements

### Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
plotly>=5.15.0
google-generativeai>=0.3.0
joblib>=1.3.0
```

### Hardware Specifications
- **Minimum**: 4GB RAM, 1GB storage
- **Recommended**: 8GB RAM, 2GB storage
- **Processing**: Multi-core CPU recommended for model training

## Installation Protocol

### Environment Setup
```bash
git clone <repository-url>
cd student-mental-health-analyzer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration Management
1. **Dataset Preparation**: Place training dataset as `Depression Student Dataset.csv`
2. **API Configuration**: Set environment variable `GEMINI_API_KEY` or modify source
3. **Model Training**: Execute initial training pipeline

### Required Data Schema
```
Features:
- Gender: categorical
- Age: numeric (16-35)
- Academic Pressure: numeric (1.0-5.0)
- Study Satisfaction: numeric (1.0-5.0)
- Sleep Duration: categorical
- Dietary Habits: categorical
- Study Hours: numeric (0-16)
- Financial Stress: numeric (1.0-5.0)
- Family History of Mental Illness: binary
- Have you ever had suicidal thoughts: binary

Target:
- Depression: binary classification
```

## Implementation Guide

### Model Training
```bash
python model_training.py
```
Generates `all_models.pkl` containing:
- Trained model objects
- Feature encoders
- Scaling parameters
- Performance metrics
- Cross-validation results

### Application Deployment
```bash
streamlit run app.py
```
Access via `http://localhost:8501`

## Performance Metrics

### Model Evaluation Framework
- **Cross-Validation**: 5-fold stratified sampling
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Selection Criteria**: Highest cross-validation accuracy
- **Risk Stratification**: Probabilistic thresholding (0.4, 0.7)

### Expected Performance Ranges
- **Ensemble Methods**: 85-92% accuracy
- **Linear Models**: 80-85% accuracy
- **Distance-Based**: 75-82% accuracy
- **Probabilistic**: 74-80% accuracy

## Data Processing Pipeline

### Preprocessing Steps
1. **Categorical Encoding**: Label encoding for nominal variables
2. **Numerical Transformation**: Sleep duration text-to-numeric conversion
3. **Feature Scaling**: StandardScaler for distance-based algorithms
4. **Missing Value Handling**: Forward-fill and default value imputation
5. **Stratified Sampling**: Maintained class distribution in train/test splits

### Feature Engineering
- Sleep duration normalized to numeric hours
- Categorical variables encoded with class preservation
- Standardization applied selectively by algorithm requirements

## Security and Privacy

### Data Handling
- **Session-Based Processing**: No persistent storage of user inputs
- **Local Computation**: All ML operations performed locally
- **API Security**: HTTPS encryption for external AI calls
- **Memory Management**: Automatic cleanup of sensitive data

### Privacy Controls
- No user identification collection
- Temporary data processing only
- Secure API key management
- GDPR-compliant data handling

## API Documentation

### Model Interface
```python
# Initialize model
model = AdvancedStudentHealthModel()

# Train on dataset
results = model.train_all_models('dataset.csv')

# Make prediction
prediction = model.predict(user_input_dict)
```

### Response Format
```python
{
    'prediction': str,              # 'Depression' or 'No Depression'
    'confidence': float,            # Probability confidence
    'risk_level': str,              # 'Low', 'Medium', 'High'
    'probabilities': dict,          # Class probabilities
    'best_model_used': str         # Selected algorithm name
}
```

## Quality Assurance

### Testing Protocol
- Unit tests for all core functions
- Integration tests for complete pipeline
- Performance benchmarks across datasets
- Cross-platform compatibility verification

### Validation Framework
- K-fold cross-validation
- Holdout test set evaluation
- Statistical significance testing
- Bias and fairness assessment

## Deployment Considerations

### Production Environment
- Container orchestration compatibility
- Horizontal scaling capabilities
- Load balancing support
- Health check endpoints

### Monitoring and Logging
- Model performance tracking
- Error rate monitoring
- Usage analytics
- System resource utilization

## Maintenance and Updates

### Model Retraining
- Automated retraining pipelines
- Performance degradation detection
- New data integration protocols
- Version control for model artifacts

### Software Updates
- Dependency vulnerability scanning
- API compatibility maintenance
- Feature enhancement integration
- Bug fix deployment procedures

## Research Applications

### Academic Use Cases
- Mental health research studies
- Algorithm comparison analysis
- Feature importance investigation
- Bias detection in ML models

### Ethical Considerations
- Algorithmic fairness assessment
- Demographic bias evaluation
- Interpretability requirements
- Consent and transparency protocols

## Support and Maintenance

### Technical Support
- GitHub issue tracking
- Documentation wiki
- Code review processes
- Community contribution guidelines

### Professional Services
- Custom model development
- Enterprise deployment support
- Training and consultation
- Compliance assistance

## License and Attribution

This project is released under the MIT License. Commercial use requires appropriate attribution and compliance with all applicable regulations regarding medical software.

## References

1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, 2011
2. Streamlit: The fastest way to build data apps, Streamlit Inc.
3. Google AI Platform: Generative AI APIs and Services
4. World Health Organization: Mental Health Guidelines and Standards

---

**Professional Notice**: This system is designed for research and educational applications. Clinical implementation requires appropriate validation studies, regulatory approval, and professional medical oversight.
