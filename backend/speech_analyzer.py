"""
Speech-to-Text and Clarity Analysis using OpenAI Whisper
This module handles:
1. Real-time audio capture and processing
2. Speech-to-text conversion using Whisper
3. Speech clarity analysis using various metrics
"""

import whisper
import numpy as np
import tempfile
import wave
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
import threading
import queue

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class SpeechAnalysisResult:
    """Results from speech analysis"""
    transcribed_text: str
    clarity_score: float  # 0-100
    fluency_score: float  # 0-100
    pace_wpm: float  # Words per minute
    filler_words_count: int
    # Map of filler word -> count, for detailed reporting
    filler_words_detail: dict
    confidence: float  # Whisper's confidence
    pronunciation_score: float  # 0-100
    
    def to_dict(self):
        return {
            "transcribed_text": self.transcribed_text,
            "clarity_score": round(self.clarity_score, 2),
            "fluency_score": round(self.fluency_score, 2),
            "pace_wpm": round(self.pace_wpm, 1),
            "filler_words_count": self.filler_words_count,
            "confidence": round(self.confidence, 2),
            "pronunciation_score": round(self.pronunciation_score, 2),
            "filler_words_detail": self.filler_words_detail
        }


class SpeechAnalyzer:
    """
    Analyzes speech for clarity, fluency, and other metrics.
    Uses OpenAI Whisper for transcription and various NLP techniques for analysis.
    """
    
    # Common filler words and phrases to detect
    FILLER_WORDS = {
        'um', 'uh', 'like', 'you know', 'so', 'basically', 'actually',
        'literally', 'right', 'okay', 'well', 'i mean', 'sort of',
        'kind of', 'you see', 'er', 'ah', 'hmm', 'mhm', 'yeah', 'man',
        'you know', 'i think', 'i suppose', 'i guess', 'and stuff',
        'or something', 'or whatever', 'at the end of the day', 'frankly',
        'honestly', 'truthfully', 'basically speaking', 'in my opinion',
        'to be honest', 'to be fair', 'if you will', 'as it were'
    }
    
    # Regex patterns for common filler word variations (handles repeated characters)
    FILLER_PATTERNS = [
        (r'\b(um+|umm+)\b', 'um'),  # um, umm, ummm, etc.
        (r'\b(uh+|uhh+)\b', 'uh'),  # uh, uhh, uhhh, etc.
        (r'\b(ah+|ahh+)\b', 'ah'),  # ah, ahh, ahhh, etc.
        (r'\b(er+|err+)\b', 'er'),  # er, err, errr, etc.
        (r'\b(like)\b', 'like'),
        (r'\b(you\s+know)\b', 'you know'),
        (r'\b(so)\b', 'so'),
        (r'\b(basically)\b', 'basically'),
        (r'\b(actually)\b', 'actually'),
        (r'\b(literally)\b', 'literally'),
        (r'\b(right)\b', 'right'),
        (r'\b(okay|ok)\b', 'okay'),
        (r'\b(well)\b', 'well'),
        (r'\b(i\s+mean)\b', 'i mean'),
        (r'\b(sort\s+of)\b', 'sort of'),
        (r'\b(kind\s+of)\b', 'kind of'),
        (r'\b(you\s+see)\b', 'you see'),
        (r'\b(hmm+)\b', 'hmm'),
        (r'\b(mhm)\b', 'mhm'),
        (r'\b(yeah)\b', 'yeah'),
        (r'\b(man)\b', 'man'),  # Common slang filler
        (r'\b(dude)\b', 'dude'),  # Common slang filler
        (r'\b(bro)\b', 'bro'),  # Common slang filler
        (r'\b(i\s+think)\b', 'i think'),
        (r'\b(i\s+suppose)\b', 'i suppose'),
        (r'\b(i\s+guess)\b', 'i guess'),
        (r'\b(to\s+be\s+honest)\b', 'to be honest'),
        (r'\b(to\s+be\s+fair)\b', 'to be fair'),
        (r'\b(honestly)\b', 'honestly'),
        (r'\b(frankly)\b', 'frankly'),
    ]
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the speech analyzer.
        
        Args:
            model_size: Whisper model size - "tiny", "base", "small", "medium", "large"
                       Smaller models are faster but less accurate.
        """
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded successfully!")
        
        # Reference vocabulary for professional speech
        self.professional_vocabulary = self._load_professional_vocabulary()
        
    def _load_professional_vocabulary(self) -> set:
        """Load a set of professional/formal vocabulary words"""
        # Common professional words used in interviews
        return {
            'experience', 'project', 'team', 'managed', 'developed',
            'implemented', 'solution', 'problem', 'challenge', 'opportunity',
            'leadership', 'collaboration', 'communication', 'technical',
            'strategic', 'initiative', 'objective', 'achievement', 'result',
            'skill', 'capability', 'responsibility', 'deadline', 'priority',
            'stakeholder', 'requirement', 'specification', 'documentation',
            'methodology', 'framework', 'architecture', 'design', 'analysis',
            'optimization', 'performance', 'quality', 'efficiency', 'productivity'
        }
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
        """
        Transcribe audio data using Whisper.
        
        Args:
            audio_data: numpy array of audio samples (float32, normalized to [-1, 1])
            sample_rate: Sample rate of the audio (default 16000 for Whisper)
            
        Returns:
            Tuple of (transcribed text, average confidence/probability)
        """
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Transcribe using Whisper
        result = self.model.transcribe(
            audio_data,
            language='en',
            task='transcribe',
            verbose=False
        )
        
        text = result.get('text', '').strip()
        
        # Calculate average confidence from segments
        segments = result.get('segments', [])
        if segments:
            avg_confidence = np.mean([
                seg.get('avg_logprob', -1.0) for seg in segments
            ])
            # Convert log probability to a 0-1 confidence score
            confidence = np.exp(avg_confidence)
            confidence = min(max(confidence, 0.0), 1.0)
        else:
            confidence = 0.5  # Default if no segments
            
        return text, confidence
    
    def transcribe_audio_file(self, file_path: str) -> Tuple[str, float]:
        """
        Transcribe audio from a file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (transcribed text, confidence)
        """
        result = self.model.transcribe(file_path, language='en', verbose=False)
        text = result.get('text', '').strip()
        
        segments = result.get('segments', [])
        if segments:
            avg_confidence = np.mean([
                seg.get('avg_logprob', -1.0) for seg in segments
            ])
            confidence = min(max(np.exp(avg_confidence), 0.0), 1.0)
        else:
            confidence = 0.5
            
        return text, confidence
    
    def count_filler_words(self, text: str) -> int:
        """Count filler words in the transcribed text using pattern matching"""
        text_lower = text.lower()
        count = 0
        
        for pattern, _ in self.FILLER_PATTERNS:
            matches = re.findall(pattern, text_lower)
            count += len(matches)
            
        return count

    def get_filler_words_detail(self, text: str) -> dict:
        """Return a dictionary of filler word -> count found in text"""
        text_lower = text.lower()
        detail = {}
        
        for pattern, normalized_name in self.FILLER_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Count occurrences (handle tuples from grouped patterns)
                count = len(matches)
                if count > 0:
                    detail[normalized_name] = detail.get(normalized_name, 0) + count
        
        return detail
    
    def calculate_pace(self, text: str, duration_seconds: float) -> float:
        """
        Calculate speaking pace in words per minute.
        
        Args:
            text: Transcribed text
            duration_seconds: Duration of the audio in seconds
            
        Returns:
            Words per minute
        """
        if duration_seconds <= 0:
            return 0.0
            
        words = text.split()
        word_count = len(words)
        
        wpm = (word_count / duration_seconds) * 60
        return wpm
    
    def calculate_clarity_score(self, text: str, reference_text: Optional[str] = None) -> float:
        """
        Calculate speech clarity score.
        
        Uses multiple metrics:
        1. Cosine similarity with reference text (if provided)
        2. Vocabulary sophistication
        3. Sentence structure analysis
        
        Args:
            text: Transcribed text to analyze
            reference_text: Optional reference/expected text for comparison
            
        Returns:
            Clarity score from 0 to 100
        """
        if not text or len(text.strip()) < 5:
            return 0.0
        
        scores = []
        
        # 1. Reference text similarity (if provided)
        if reference_text and len(reference_text.strip()) > 5:
            similarity = self._calculate_text_similarity(text, reference_text)
            scores.append(similarity * 100)
        
        # 2. Vocabulary analysis
        vocab_score = self._analyze_vocabulary(text)
        scores.append(vocab_score)
        
        # 3. Sentence coherence
        coherence_score = self._analyze_sentence_coherence(text)
        scores.append(coherence_score)
        
        # 4. Low filler word usage (inverse of filler density)
        words = text.split()
        if len(words) > 0:
            filler_count = self.count_filler_words(text)
            filler_density = filler_count / len(words)
            filler_score = max(0, 100 - (filler_density * 500))  # Penalize filler words
            scores.append(filler_score)
        
        return np.mean(scores) if scores else 50.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            return 0.5  # Default similarity on error
    
    def _analyze_vocabulary(self, text: str) -> float:
        """Analyze vocabulary sophistication"""
        try:
            words = word_tokenize(text.lower())
            if len(words) < 3:
                return 50.0
            
            # Unique words ratio
            unique_ratio = len(set(words)) / len(words)
            
            # Professional vocabulary usage
            professional_count = sum(1 for w in words if w in self.professional_vocabulary)
            professional_ratio = professional_count / len(words)
            
            # Average word length (longer words often indicate sophistication)
            avg_word_length = np.mean([len(w) for w in words if w.isalpha()])
            word_length_score = min(avg_word_length / 8, 1.0)  # Normalize to max of 8 chars
            
            score = (unique_ratio * 30 + professional_ratio * 40 + word_length_score * 30)
            return min(score * 100 / 30, 100)  # Normalize to 0-100
            
        except Exception:
            return 50.0
    
    def _analyze_sentence_coherence(self, text: str) -> float:
        """Analyze sentence structure and coherence"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) == 0:
                return 50.0
            
            # Average sentence length (ideal is 15-20 words)
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            
            # Score based on ideal sentence length
            if 10 <= avg_sentence_length <= 25:
                length_score = 100
            elif avg_sentence_length < 10:
                length_score = (avg_sentence_length / 10) * 100
            else:
                length_score = max(0, 100 - (avg_sentence_length - 25) * 5)
            
            # Check for sentence variety
            sentence_lengths = [len(s.split()) for s in sentences]
            if len(sentence_lengths) > 1:
                variety_score = min(np.std(sentence_lengths) * 10, 100)
            else:
                variety_score = 50
            
            return (length_score * 0.6 + variety_score * 0.4)
            
        except Exception:
            return 50.0
    
    def calculate_fluency_score(self, text: str, duration_seconds: float) -> float:
        """
        Calculate fluency score based on pace and flow.
        
        Args:
            text: Transcribed text
            duration_seconds: Duration of the audio
            
        Returns:
            Fluency score from 0 to 100
        """
        if not text or duration_seconds <= 0:
            return 0.0
        
        scores = []
        
        # 1. Speaking pace score (ideal: 120-160 WPM)
        wpm = self.calculate_pace(text, duration_seconds)
        if 120 <= wpm <= 160:
            pace_score = 100
        elif 100 <= wpm < 120 or 160 < wpm <= 180:
            pace_score = 85
        elif 80 <= wpm < 100 or 180 < wpm <= 200:
            pace_score = 70
        else:
            pace_score = max(0, 50 - abs(wpm - 140) / 2)
        scores.append(pace_score)
        
        # 2. Low filler word usage
        words = text.split()
        if len(words) > 0:
            filler_count = self.count_filler_words(text)
            filler_ratio = filler_count / len(words)
            filler_score = max(0, 100 - filler_ratio * 400)
            scores.append(filler_score)
        
        # 3. Consistent flow (sentences should be complete)
        try:
            sentences = sent_tokenize(text)
            complete_sentences = sum(1 for s in sentences if s.strip().endswith(('.', '?', '!')))
            if len(sentences) > 0:
                completion_score = (complete_sentences / len(sentences)) * 100
            else:
                completion_score = 50
            scores.append(completion_score)
        except Exception:
            pass
        
        return np.mean(scores) if scores else 50.0
    
    def calculate_pronunciation_score(self, confidence: float, text: str) -> float:
        """
        Estimate pronunciation score based on Whisper's confidence and text quality.
        
        Note: This is an estimation. True pronunciation scoring would require
        phoneme-level analysis.
        
        Args:
            confidence: Whisper's transcription confidence
            text: Transcribed text
            
        Returns:
            Pronunciation score from 0 to 100
        """
        # Base score from Whisper confidence
        base_score = confidence * 100
        
        # Adjust based on text coherence
        if text:
            # Check for garbled/nonsensical text patterns
            words = text.split()
            if len(words) > 0:
                # Average word length (very short average might indicate issues)
                avg_len = np.mean([len(w) for w in words])
                if avg_len < 2:
                    base_score *= 0.8
                    
                # Check for repeated characters (might indicate transcription issues)
                repeated_pattern = re.findall(r'(.)\1{3,}', text)
                if repeated_pattern:
                    base_score *= 0.9
        
        return min(max(base_score, 0), 100)
    
    def analyze_speech(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int = 16000,
        duration_seconds: float = None,
        reference_text: Optional[str] = None
    ) -> SpeechAnalysisResult:
        """
        Perform comprehensive speech analysis.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Audio sample rate
            duration_seconds: Duration of audio (calculated if not provided)
            reference_text: Optional reference text for comparison
            
        Returns:
            SpeechAnalysisResult with all metrics
        """
        # Calculate duration if not provided
        if duration_seconds is None:
            duration_seconds = len(audio_data) / sample_rate
        
        # Transcribe
        text, confidence = self.transcribe_audio(audio_data, sample_rate)
        
        # Calculate metrics
        clarity = self.calculate_clarity_score(text, reference_text)
        fluency = self.calculate_fluency_score(text, duration_seconds)
        pace = self.calculate_pace(text, duration_seconds)
        filler_count = self.count_filler_words(text)
        filler_detail = self.get_filler_words_detail(text)
        pronunciation = self.calculate_pronunciation_score(confidence, text)
        
        return SpeechAnalysisResult(
            transcribed_text=text,
            clarity_score=clarity,
            fluency_score=fluency,
            pace_wpm=pace,
            filler_words_count=filler_count,
            confidence=confidence,
            pronunciation_score=pronunciation,
            filler_words_detail=filler_detail
        )
    
    def analyze_speech_file(
        self,
        file_path: str,
        reference_text: Optional[str] = None
    ) -> SpeechAnalysisResult:
        """
        Analyze speech from an audio file.
        
        Args:
            file_path: Path to audio file
            reference_text: Optional reference text
            
        Returns:
            SpeechAnalysisResult
        """
        # Get duration from file
        import wave
        try:
            with wave.open(file_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
        except Exception:
            duration = 30.0  # Default duration
        
        # Transcribe
        text, confidence = self.transcribe_audio_file(file_path)
        
        # Calculate metrics
        clarity = self.calculate_clarity_score(text, reference_text)
        fluency = self.calculate_fluency_score(text, duration)
        pace = self.calculate_pace(text, duration)
        filler_count = self.count_filler_words(text)
        filler_detail = self.get_filler_words_detail(text)
        pronunciation = self.calculate_pronunciation_score(confidence, text)
        
        return SpeechAnalysisResult(
            transcribed_text=text,
            clarity_score=clarity,
            fluency_score=fluency,
            pace_wpm=pace,
            filler_words_count=filler_count,
            confidence=confidence,
            pronunciation_score=pronunciation,
            filler_words_detail=filler_detail
        )


# Singleton instance for reuse
_analyzer_instance: Optional[SpeechAnalyzer] = None


def get_speech_analyzer(model_size: str = "base") -> SpeechAnalyzer:
    """Get or create the speech analyzer singleton"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SpeechAnalyzer(model_size)
    return _analyzer_instance


if __name__ == "__main__":
    # Test the speech analyzer
    print("Testing Speech Analyzer...")
    
    analyzer = SpeechAnalyzer(model_size="base")
    
    # Create a test audio file or use existing one
    test_text = """
    In my previous role as a software engineer, I worked extensively with 
    Python and JavaScript. I led a team of five developers on a project 
    that improved system performance by 40 percent. The main challenge was 
    optimizing database queries while maintaining code readability.
    """
    
    print(f"Test text clarity score: {analyzer.calculate_clarity_score(test_text):.2f}")
    print(f"Filler words: {analyzer.count_filler_words(test_text)}")
