def get_explanation(text, label, confidence):
    """Generate a simple heuristic explanation based on the text length and keywords."""
    sensational_words = ['shocking', 'unbelievable', 'mind-blowing', 'secret', 'banned']
    found_words = [w for w in sensational_words if w in text.lower()]
    
    explanation = f"The model is {confidence:.1%} confident that this article is {label}. "
    
    if len(found_words) > 0:
        explanation += f"It detected sensationalist keywords often associated with fake news: {', '.join(found_words)}. "
        
    if len(text.split()) < 20:
        explanation += "The text is very short, which might lead to less reliable predictions."
        
    return explanation
