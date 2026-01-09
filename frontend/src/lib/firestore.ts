import { 
  collection, 
  doc, 
  addDoc, 
  updateDoc, 
  getDoc, 
  getDocs, 
  query, 
  where, 
  orderBy, 
  Timestamp,
  serverTimestamp 
} from "firebase/firestore";
import { db } from "./firebase";

// Types for test results
export interface SpeechTestResult {
  fluency: number;
  fillerWords: number;
  pace: number;
  pronunciation: number;
  recordingDuration: number;
  // Optional detailed fields from backend
  transcribedText?: string;
  clarityScore?: number;
  fillerWordsDetail?: Record<string, number>;
}

export interface InterviewQuestion {
  title: string;
  category: string;
  difficulty: string;
  question: string;
  answer: string;
  score?: number;
  feedback?: string;
}

export interface BodyLanguageMetrics {
  eyeContact: number;
  avgBlinkRate: number;
  confidenceCurve: number;
  emotionTimeline: string[];
}

export interface InterviewSession {
  id?: string;
  userId: string;
  createdAt: Timestamp | ReturnType<typeof serverTimestamp>;
  completedAt?: Timestamp | ReturnType<typeof serverTimestamp>;
  status: "in-progress" | "completed";
  
  // Speech test results
  speechTest?: SpeechTestResult;
  
  // Interview results
  questions?: InterviewQuestion[];
  liveMetrics?: {
    attention: number;
    eyeContact: number;
    blinkRate: number;
    emotion: string;
    confidence: number;
    speakingPace: number;
  };
  bodyLanguage?: BodyLanguageMetrics;
  
  // Overall scores
  overallScore?: number;
  technicalScore?: number;
  communicationScore?: number;
  bodyLanguageScore?: number;
}

// Create a new interview session
export const createInterviewSession = async (userId: string): Promise<string> => {
  const sessionsRef = collection(db, "interviewSessions");
  const docRef = await addDoc(sessionsRef, {
    userId,
    createdAt: serverTimestamp(),
    status: "in-progress"
  });
  return docRef.id;
};

// Update speech test results
export const saveSpeechTestResults = async (
  sessionId: string, 
  results: SpeechTestResult
): Promise<void> => {
  const sessionRef = doc(db, "interviewSessions", sessionId);
  await updateDoc(sessionRef, {
    speechTest: results
  });
};

// Update interview answers and metrics
export const saveInterviewResults = async (
  sessionId: string,
  questions: InterviewQuestion[],
  liveMetrics: InterviewSession["liveMetrics"]
): Promise<void> => {
  const sessionRef = doc(db, "interviewSessions", sessionId);
  await updateDoc(sessionRef, {
    questions,
    liveMetrics
  });
};

// Complete the interview session with final scores
export const completeInterviewSession = async (
  sessionId: string,
  finalResults: {
    overallScore: number;
    technicalScore: number;
    communicationScore: number;
    bodyLanguageScore: number;
    bodyLanguage: BodyLanguageMetrics;
    questions: InterviewQuestion[];
  }
): Promise<void> => {
  const sessionRef = doc(db, "interviewSessions", sessionId);
  await updateDoc(sessionRef, {
    ...finalResults,
    status: "completed",
    completedAt: serverTimestamp()
  });
};

// Get a specific interview session
export const getInterviewSession = async (sessionId: string): Promise<InterviewSession | null> => {
  const sessionRef = doc(db, "interviewSessions", sessionId);
  const snapshot = await getDoc(sessionRef);
  
  if (snapshot.exists()) {
    return { id: snapshot.id, ...snapshot.data() } as InterviewSession;
  }
  return null;
};

// Get all sessions for a user
export const getUserInterviewSessions = async (userId: string): Promise<InterviewSession[]> => {
  const sessionsRef = collection(db, "interviewSessions");
  const q = query(
    sessionsRef, 
    where("userId", "==", userId),
    orderBy("createdAt", "desc")
  );
  
  const snapshot = await getDocs(q);
  return snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as InterviewSession));
};

// Get the latest session for a user
export const getLatestSession = async (userId: string): Promise<InterviewSession | null> => {
  const sessions = await getUserInterviewSessions(userId);
  return sessions.length > 0 ? sessions[0] : null;
};
