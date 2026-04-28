"""
Custom NLP utility functions for Assignment 4 Part 2.
These functions perform basic text processing tasks and are designed to be
embedded and retrieved via the Retriever system.
"""

import re
import string

def clean_text(text: str) -> str:
    """
    Cleans the input text by lowercasing, removing punctuation, 
    and collapsing repeated whitespace into a single space.
    """
    # Lowercase the text
    text = text.lower()
    # Remove punctuation using regular expressions
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_word_frequencies(text: str) -> dict[str, int]:
    """
    Cleans and tokenizes text, returning a dictionary mapping each word
    to its frequency count.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    
    freq_dict = {}
    for word in words:
        freq_dict[word] = freq_dict.get(word, 0) + 1
        
    return freq_dict


def extract_hashtags(text: str) -> list[str]:
    """
    Extracts hashtag words from social media-style text.
    Returns them as a list of strings without the # symbol.
    """
    # Find all words that start with # and consist of word characters
    matches = re.findall(r"#(\w+)", text)
    return matches


def simple_sentiment_score(text: str) -> dict[str, int | str]:
    """
    Calculates a basic sentiment score using small sets of positive and negative words.
    Returns the counts and an overall sentiment classification.
    """
    positive_words = {"good", "great", "excellent", "happy", "love", "awesome", "perfect"}
    negative_words = {"bad", "terrible", "awful", "sad", "hate", "worst", "poor"}
    
    words = clean_text(text).split()
    
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    if pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        # A tie or purely 0 match logic
        sentiment = "neutral"
        
    return {
        "positive_count": pos_count,
        "negative_count": neg_count,
        "sentiment": sentiment
    }


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using simple punctuation rules (e.g., periods, 
    exclamation marks, question marks) and returns a list of stripped strings.
    """
    # Split using a regex that identifies sentence-ending punctuation followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    
    # Strip whitespace and filter out any completely empty strings
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


if __name__ == "__main__":
    sample_text = "I absolutely love #NLP and #Python! Wait, is this terrible? No, it's great    text...   "
    
    print("Testing custom NLP functions:\n")
    
    print("1. clean_text:")
    print("  Input:", repr(sample_text))
    print("  Output:", repr(clean_text(sample_text)))
    print()
    
    print("2. count_word_frequencies:")
    print("  Output:", count_word_frequencies(sample_text))
    print()
    
    print("3. extract_hashtags:")
    print("  Output:", extract_hashtags(sample_text))
    print()
    
    print("4. simple_sentiment_score:")
    print("  Output:", simple_sentiment_score(sample_text))
    print()
    
    print("5. split_into_sentences:")
    print("  Output:", split_into_sentences(sample_text))
