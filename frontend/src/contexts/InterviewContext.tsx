import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { 
  createInterviewSession, 
  saveSpeechTestResults, 
  saveInterviewResults,
  completeInterviewSession,
  getInterviewSession,
  type InterviewSession,
  type SpeechTestResult,
  type InterviewQuestion,
  type BodyLanguageMetrics
} from "@/lib/firestore";
import { useAuth } from "./AuthContext";

interface InterviewContextType {
  sessionId: string | null;
  session: InterviewSession | null;
  isLoading: boolean;
  
  // Session management
  startNewSession: () => Promise<string>;
  loadSession: (sessionId: string) => Promise<void>;
  
  // Save results
  saveSpeechTest: (results: SpeechTestResult) => Promise<void>;
  saveInterview: (questions: InterviewQuestion[], liveMetrics: InterviewSession["liveMetrics"]) => Promise<void>;
  completeSession: (finalResults: {
    overallScore: number;
    technicalScore: number;
    communicationScore: number;
    bodyLanguageScore: number;
    bodyLanguage: BodyLanguageMetrics;
    questions: InterviewQuestion[];
  }) => Promise<void>;
  
  // Local state for in-progress data
  speechTestResults: SpeechTestResult | null;
  setSpeechTestResults: (results: SpeechTestResult) => void;
  interviewAnswers: string[];
  setInterviewAnswers: (answers: string[]) => void;
}

const InterviewContext = createContext<InterviewContextType | null>(null);

export const useInterview = () => {
  const context = useContext(InterviewContext);
  if (!context) {
    throw new Error("useInterview must be used within an InterviewProvider");
  }
  return context;
};

interface InterviewProviderProps {
  children: ReactNode;
}

export const InterviewProvider = ({ children }: InterviewProviderProps) => {
  const { currentUser } = useAuth();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // Local state for in-progress interview
  const [speechTestResults, setSpeechTestResults] = useState<SpeechTestResult | null>(null);
  const [interviewAnswers, setInterviewAnswers] = useState<string[]>(["", "", ""]);

  // Bootstrap from localStorage (supports non-auth flows and page reloads)
  useEffect(() => {
    try {
      const storedSessionId = localStorage.getItem("interview_session_id");
      const storedSpeech = localStorage.getItem("speech_test_results");
      const storedResults = localStorage.getItem("interview_results");
      
      if (storedSessionId) {
        setSessionId(storedSessionId);
      }
      if (storedSpeech) {
        setSpeechTestResults(JSON.parse(storedSpeech));
      }
      if (storedResults && !session) {
        // Load interview results as session data for non-logged-in users
        const results = JSON.parse(storedResults);
        setSession({
          ...results,
          userId: "guest",
          createdAt: new Date() as any,
          status: "completed"
        });
      }
    } catch (e) {
      console.warn("Failed to load from localStorage:", e);
    }
  }, []);

  const startNewSession = async (): Promise<string> => {
    if (!currentUser) throw new Error("User must be logged in");
    
    setIsLoading(true);
    try {
      const newSessionId = await createInterviewSession(currentUser.uid);
      setSessionId(newSessionId);
      try { localStorage.setItem("interview_session_id", newSessionId); } catch {}
      setSpeechTestResults(null);
      setInterviewAnswers(["", "", ""]);
      return newSessionId;
    } finally {
      setIsLoading(false);
    }
  };

  const loadSession = async (id: string): Promise<void> => {
    setIsLoading(true);
    try {
      const loadedSession = await getInterviewSession(id);
      if (loadedSession) {
        setSession(loadedSession);
        setSessionId(id);
        if (loadedSession.speechTest) {
          setSpeechTestResults(loadedSession.speechTest);
        }
        if (loadedSession.questions) {
          setInterviewAnswers(loadedSession.questions.map(q => q.answer));
        }
      }
    } finally {
      setIsLoading(false);
    }
  };

  const saveSpeechTest = async (results: SpeechTestResult): Promise<void> => {
    if (!sessionId) {
      // Create a new session if one doesn't exist
      if (currentUser) {
        const newSessionId = await startNewSession();
        await saveSpeechTestResults(newSessionId, results);
      }
    } else {
      await saveSpeechTestResults(sessionId, results);
    }
    setSpeechTestResults(results);
    try { localStorage.setItem("speech_test_results", JSON.stringify(results)); } catch {}
  };

  const saveInterview = async (
    questions: InterviewQuestion[], 
    liveMetrics: InterviewSession["liveMetrics"]
  ): Promise<void> => {
    if (!sessionId) throw new Error("No active session");
    await saveInterviewResults(sessionId, questions, liveMetrics);
  };

  const completeSession = async (finalResults: {
    overallScore: number;
    technicalScore: number;
    communicationScore: number;
    bodyLanguageScore: number;
    bodyLanguage: BodyLanguageMetrics;
    questions: InterviewQuestion[];
  }): Promise<void> => {
    if (!sessionId) {
      // If user is logged in, create a new session and save
      if (currentUser) {
        const newSessionId = await startNewSession();
        await completeInterviewSession(newSessionId, finalResults);
        setSessionId(newSessionId);
        await loadSession(newSessionId);
      } else {
        // For non-logged-in users, just store in memory and localStorage
        setSpeechTestResults((prev) => prev || null);
        setInterviewAnswers(finalResults.questions.map(q => q.answer));
        // Store session data locally
        try {
          localStorage.setItem("interview_results", JSON.stringify(finalResults));
        } catch {}
      }
    } else {
      await completeInterviewSession(sessionId, finalResults);
      // Reload the session to get updated data
      await loadSession(sessionId);
    }
  };

  const value: InterviewContextType = {
    sessionId,
    session,
    isLoading,
    startNewSession,
    loadSession,
    saveSpeechTest,
    saveInterview,
    completeSession,
    speechTestResults,
    setSpeechTestResults,
    interviewAnswers,
    setInterviewAnswers
  };

  return (
    <InterviewContext.Provider value={value}>
      {children}
    </InterviewContext.Provider>
  );
};
