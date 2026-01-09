import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Download, RotateCcw, Home, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useInterview } from "@/contexts/InterviewContext";
import { useAuth } from "@/contexts/AuthContext";
import { generateInterviewReport } from "@/lib/pdfGenerator";

const Report = () => {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const { session, sessionId, loadSession, speechTestResults } = useInterview();
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);
  const [isUsingMockData, setIsUsingMockData] = useState(false);

  useEffect(() => {
    // Load session data if we have a sessionId but no session loaded
    if (sessionId && !session) {
      setIsLoadingSession(true);
      loadSession(sessionId).finally(() => setIsLoadingSession(false));
    }
  }, [sessionId, session, loadSession]);

  // Determine if we're using mock data
  const hasRealData = session || speechTestResults || sessionId;
  
  // Mock data for fallback
  const mockData = {
    overallScore: 82,
    technicalScore: 78,
    communicationScore: 85,
    bodyLanguageScore: 80,
    speechTest: {
      fluency: 82,
      pronunciation: 88,
      fillerWords: 3,
      pace: 145,
      recordingDuration: 45,
      transcribedText: "This is a sample transcription of your speech. It demonstrates how your actual interview response would appear here with detailed analysis.",
      clarityScore: 85,
      fillerWordsDetail: { um: 1, uh: 1, like: 1 }
    },
    bodyLanguage: {
      eyeContact: 75,
      avgBlinkRate: 15,
      confidenceCurve: 82,
      emotionTimeline: ["Confident", "Focused", "Calm"]
    },
    questions: [
      {
        title: "Core Concepts",
        category: "Java",
        difficulty: "Medium",
        question: "Explain the difference between an interface and an abstract class in Java.",
        answer: "Interfaces define a contract for what a class should do, while abstract classes provide a base implementation. Interfaces support multiple inheritance of type while abstract classes don't.",
        score: 85,
        feedback: "Strong understanding of fundamental concepts. Clear explanation of interface vs abstract class with practical examples."
      },
      {
        title: "Problem Solving",
        category: "Algorithms",
        difficulty: "Hard",
        question: "Design an algorithm to find the longest palindromic substring.",
        answer: "We can use dynamic programming to solve this in O(n²) time and O(n²) space by building a table to track palindromic substrings.",
        score: 75,
        feedback: "Good approach to the problem. Consider optimizing time complexity using dynamic programming or expanding around centers approach."
      },
      {
        title: "System Design",
        category: "Architecture",
        difficulty: "Hard",
        question: "How would you design a URL shortening service like bit.ly?",
        answer: "Use a database to map short codes to long URLs, implement caching with Redis, use consistent hashing for distribution, and include analytics for tracking.",
        score: 74,
        feedback: "Solid high-level design. Could improve on discussing caching strategies, database sharding, and handling edge cases."
      }
    ]
  };

  // Use session data if available, otherwise use speech test results, otherwise use mock data
  useEffect(() => {
    if (!session && !speechTestResults && !sessionId) {
      setIsUsingMockData(true);
    } else {
      setIsUsingMockData(false);
    }
  }, [session, speechTestResults, sessionId]);

  // Use real data or mock data
  const data = session ? {
    overallScore: session.overallScore || mockData.overallScore,
    technicalScore: session.technicalScore || mockData.technicalScore,
    communicationScore: session.communicationScore || mockData.communicationScore,
    bodyLanguageScore: session.bodyLanguageScore || mockData.bodyLanguageScore,
    speechTest: session.speechTest || mockData.speechTest,
    bodyLanguage: session.bodyLanguage || mockData.bodyLanguage,
    questions: session.questions || mockData.questions
  } : {
    overallScore: 82,
    technicalScore: 78,
    communicationScore: 85,
    bodyLanguageScore: 80,
    speechTest: mockData.speechTest,
    bodyLanguage: mockData.bodyLanguage,
    questions: mockData.questions
  };

  // Use session data if available, otherwise use defaults
  const overallScore = data.overallScore;
  const scores = {
    technical: data.technicalScore,
    communication: data.communicationScore,
    bodyLanguage: data.bodyLanguageScore
  };

  const speechMetrics = data.speechTest;

  const bodyLanguageMetrics = data.bodyLanguage;

  const questionResults = data.questions;

  const handleDownloadPDF = async () => {
    setIsGeneratingPDF(true);
    try {
      // Build the session object for PDF generation
      const pdfSession = {
        id: sessionId || undefined,
        userId: currentUser?.uid || "anonymous",
        createdAt: new Date() as any,
        status: "completed" as const,
        overallScore,
        technicalScore: scores.technical,
        communicationScore: scores.communication,
        bodyLanguageScore: scores.bodyLanguage,
        speechTest: {
          fluency: speechMetrics.fluency,
          pronunciation: speechMetrics.pronunciation,
          fillerWords: speechMetrics.fillerWords,
          pace: speechMetrics.pace,
          recordingDuration: speechMetrics.recordingDuration || 0
        },
        bodyLanguage: bodyLanguageMetrics,
        questions: questionResults.map(q => ({
          title: q.title,
          category: q.category,
          difficulty: q.difficulty,
          question: q.question,
          answer: q.answer || "",
          score: q.score,
          feedback: q.feedback
        }))
      };
      
      generateInterviewReport(pdfSession, currentUser?.displayName || currentUser?.email || undefined);
      toast.success("PDF report downloaded successfully!");
    } catch (error) {
      console.error("Failed to generate PDF:", error);
      toast.error("Failed to generate PDF report");
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  const ScoreCard = ({ label, score, color }: { label: string; score: number; color: string }) => (
    <Card>
      <CardContent className="pt-6 text-center">
        <p className="text-sm text-muted-foreground mb-2">{label}</p>
        <p className={`text-4xl font-bold font-heading ${color}`}>{score}</p>
        <Progress value={score} className="mt-3 h-2" />
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-heading font-bold text-primary">Interview Assessment Report</h1>
          <Button variant="outline" onClick={() => navigate("/")}>
            <Home className="mr-2 h-4 w-4" />
            Exit
          </Button>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-5xl space-y-8">
        {/* Mock Data Banner */}
        {isUsingMockData && (
          <Card className="border-yellow-500/50 bg-yellow-50/50 dark:bg-yellow-950/20">
            <CardContent className="pt-4">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                ⚠️ <strong>Demo Mode:</strong> You're viewing a sample report. Complete the full interview to generate your personalized results!
              </p>
            </CardContent>
          </Card>
        )}
        {/* Overall Score */}
        <Card className="border-accent/20 bg-gradient-primary text-primary-foreground">
          <CardContent className="pt-8 pb-8 text-center">
            <h2 className="text-2xl font-heading font-semibold mb-2">Your Interview Performance</h2>
            <div className="text-7xl font-bold font-heading my-6">{overallScore}/100</div>
            <Badge className="bg-primary-foreground/20 text-primary-foreground hover:bg-primary-foreground/30">
              Above Average Performance
            </Badge>
          </CardContent>
        </Card>

        {/* Score Breakdown */}
        <div className="grid md:grid-cols-3 gap-4">
          <ScoreCard label="Technical Score" score={scores.technical} color="text-primary" />
          <ScoreCard label="Communication" score={scores.communication} color="text-accent" />
          <ScoreCard label="Body Language" score={scores.bodyLanguage} color="text-success" />
        </div>

        {/* Speech Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🎤 Speech Analysis
            </CardTitle>
            <CardDescription>Your communication metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium">Fluency Score</span>
                  <span className="text-sm font-bold">{speechMetrics.fluency}/100</span>
                </div>
                <Progress value={speechMetrics.fluency} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium">Pronunciation</span>
                  <span className="text-sm font-bold">{speechMetrics.pronunciation}/100</span>
                </div>
                <Progress value={speechMetrics.pronunciation} className="h-2" />
              </div>
            </div>
            
            <div className="grid md:grid-cols-2 gap-4 pt-4">
              <Card className="bg-muted">
                <CardContent className="pt-4">
                  <p className="text-2xl font-bold">{speechMetrics.fillerWords}</p>
                  <p className="text-sm text-muted-foreground">Filler words detected</p>
                  {speechMetrics.fillerWords > 0 && (
                    <details className="mt-3 text-sm">
                      <summary className="cursor-pointer font-medium text-primary hover:underline">
                        View filler words used
                      </summary>
                      <div className="mt-2 space-y-1 text-muted-foreground pl-4 border-l-2 border-accent">
                        {Object.keys(speechMetrics.fillerWordsDetail || {}).length > 0 ? (
                          Object.entries(speechMetrics.fillerWordsDetail!)
                            .sort((a, b) => b[1] - a[1])
                            .map(([word, count]) => (
                              <div key={word} className="flex items-center justify-between">
                                <span>{word}</span>
                                <span className="font-medium">{count}</span>
                              </div>
                            ))
                        ) : (
                          <p>No detailed filler words available</p>
                        )}
                      </div>
                    </details>
                  )}
                </CardContent>
              </Card>
              <Card className="bg-muted">
                <CardContent className="pt-4">
                  <p className="text-2xl font-bold">{speechMetrics.pace} WPM</p>
                  <p className="text-sm text-muted-foreground">Speaking pace</p>
                </CardContent>
              </Card>
            </div>

            {/* Transcript viewer */}
            {speechMetrics.transcribedText && (
              <div className="pt-4">
                <details className="text-sm">
                  <summary className="cursor-pointer font-medium text-primary hover:underline">
                    View Speech Transcript
                  </summary>
                  <p className="mt-2 text-muted-foreground whitespace-pre-wrap pl-4 border-l-2 border-accent">
                    {speechMetrics.transcribedText}
                  </p>
                </details>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Body Language Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              👁️ Body Language Analysis
            </CardTitle>
            <CardDescription>Non-verbal communication assessment</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-3 gap-4">
              <Card className="bg-muted">
                <CardContent className="pt-4">
                  <p className="text-2xl font-bold">{bodyLanguageMetrics.eyeContact}%</p>
                  <p className="text-sm text-muted-foreground">Eye contact maintained</p>
                </CardContent>
              </Card>
              <Card className="bg-muted">
                <CardContent className="pt-4">
                  <p className="text-2xl font-bold">{bodyLanguageMetrics.avgBlinkRate}/min</p>
                  <p className="text-sm text-muted-foreground">Average blink rate</p>
                </CardContent>
              </Card>
              <Card className="bg-muted">
                <CardContent className="pt-4">
                  <p className="text-2xl font-bold">{bodyLanguageMetrics.confidenceCurve}%</p>
                  <p className="text-sm text-muted-foreground">Confidence score</p>
                </CardContent>
              </Card>
            </div>
            
            <div className="pt-4">
              <p className="text-sm font-medium mb-2">Emotion Timeline</p>
              <div className="flex gap-2">
                {bodyLanguageMetrics.emotionTimeline.map((emotion, idx) => (
                  <Badge key={idx} variant="outline">{emotion}</Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Question-by-Question Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              📝 Technical Answers Review
            </CardTitle>
            <CardDescription>Detailed feedback on your responses</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {questionResults.map((result, idx) => (
              <div key={idx}>
                {idx > 0 && <Separator className="my-6" />}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold">Question {idx + 1}: {result.title}</h4>
                    <Badge variant={(result.score || 0) >= 80 ? "default" : "secondary"}>
                      {result.score}/100
                    </Badge>
                  </div>
                  
                  <div className="bg-muted p-4 rounded-lg">
                    <p className="text-sm font-medium mb-1">AI Feedback:</p>
                    <p className="text-sm text-muted-foreground">{result.feedback}</p>
                  </div>
                  
                  {result.answer && (
                    <details className="text-sm">
                      <summary className="cursor-pointer font-medium text-primary hover:underline">
                        View Your Answer
                      </summary>
                      <p className="mt-2 text-muted-foreground pl-4 border-l-2 border-accent">
                        {result.answer}
                      </p>
                    </details>
                  )}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-4 justify-center">
          <Button size="lg" onClick={handleDownloadPDF} className="gap-2" disabled={isGeneratingPDF}>
            {isGeneratingPDF ? <Loader2 className="h-5 w-5 animate-spin" /> : <Download className="h-5 w-5" />}
            Download PDF Report
          </Button>
          <Button size="lg" variant="outline" onClick={() => navigate("/")} className="gap-2">
            <RotateCcw className="h-5 w-5" />
            Retry Full Test
          </Button>
          <Button size="lg" variant="outline" onClick={() => navigate("/speech-test")} className="gap-2">
            <RotateCcw className="h-5 w-5" />
            Retake Speech Test
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Report;
