import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, Eye, Activity, Smile, Clock, Loader2, Mic, MicOff, Video, VideoOff, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { useInterview } from "@/contexts/InterviewContext";
import { useAuth } from "@/contexts/AuthContext";
import { useInterviewMetrics, VideoMetrics, AudioMetrics } from "@/hooks/useInterviewMetrics";

const Interview = () => {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const { completeSession, speechTestResults } = useInterview();
  
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<string[]>(["", "", ""]);
  const [timeRemaining, setTimeRemaining] = useState(1800); // 30 minutes
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  // Video element ref
  const videoElementRef = useRef<HTMLVideoElement>(null);
  
  // Media recorder for audio
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  
  // Use our custom hook for real-time metrics
  const { 
    metrics, 
    startVideo, 
    stopVideo, 
    sendAudioForAnalysis 
  } = useInterviewMetrics();

  const questions = [
    {
      title: "Core Concepts",
      difficulty: "Medium",
      question: "Explain the difference between an interface and an abstract class in Java. When would you use one over the other?",
      category: "Java"
    },
    {
      title: "Problem Solving",
      difficulty: "Hard",
      question: "Design an algorithm to find the longest palindromic substring in a given string. What is the time complexity?",
      category: "Algorithms"
    },
    {
      title: "System Design",
      difficulty: "Hard",
      question: "How would you design a URL shortening service like bit.ly? Discuss the key components and scalability considerations.",
      category: "Architecture"
    }
  ];

  // Compute live metrics from backend - use 0 as fallback until real data arrives
  const liveMetrics = {
    attention: metrics.video?.attention_score ?? 0,
    eyeContact: metrics.video?.eye_contact_score ?? 0,
    isLooking: metrics.video?.is_looking_at_camera ?? false,
    faceDetected: metrics.video?.face_detected ?? false,
    yaw: metrics.video?.yaw ?? 0,
    pitch: metrics.video?.pitch ?? 0,
    roll: metrics.video?.roll ?? 0,
    // Audio metrics
    clarity: metrics.audio?.clarity_score ?? 0,
    fluency: metrics.audio?.fluency_score ?? 0,
    pace: metrics.audio?.pace_wpm ?? 0,
    fillerWords: metrics.audio?.filler_words_count ?? 0,
    transcribedText: metrics.audio?.transcribed_text ?? "",
    pronunciation: metrics.audio?.pronunciation_score ?? 0,
    // Overall confidence based on all metrics
    confidence: metrics.isConnected ? Math.round(
      ((metrics.video?.attention_score ?? 0) + 
       (metrics.video?.eye_contact_score ?? 0) + 
       (metrics.audio?.clarity_score ?? 0)) / 3
    ) : 0,
    emotion: metrics.video?.is_looking_at_camera ? "Focused" : metrics.video?.face_detected ? "Distracted" : "No Face",
    blinkRate: 15,
    speakingPace: metrics.audio?.pace_wpm ?? 0
  };

  // Start video when component mounts
  useEffect(() => {
    const initializeVideo = async () => {
      if (videoElementRef.current) {
        try {
          await startVideo(videoElementRef.current);
          toast.success("Camera connected successfully");
        } catch (error) {
          toast.error("Failed to access camera. Please check permissions.");
        }
      }
    };

    initializeVideo();

    return () => {
      stopVideo();
      stopRecording();
    };
  }, []);

  // Timer countdown
  useEffect(() => {
    const timer = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 0) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Start audio recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Convert to WAV for Whisper (better compatibility)
        try {
          toast.info("Analyzing your speech...");
          await sendAudioForAnalysis(audioBlob, questions[currentQuestion].question);
          toast.success("Speech analysis complete");
        } catch (error) {
          toast.error("Failed to analyze speech");
        }
        
        // Stop audio stream
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(1000); // Collect data every second
      setIsRecording(true);
      toast.success("Recording started - speak your answer");
    } catch (error) {
      toast.error("Failed to access microphone");
    }
  };

  // Stop audio recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Toggle recording
  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleAnswerChange = (value: string) => {
    const newAnswers = [...answers];
    newAnswers[currentQuestion] = value;
    setAnswers(newAnswers);
  };

  const handleNextQuestion = async () => {
    // Stop recording before moving to next question
    if (isRecording) {
      stopRecording();
    }

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      toast.success("Moving to next question");
    } else {
      // Finish interview - save all results
      setIsSubmitting(true);
      try {
        // Generate scores based on answer length and real metrics from backend
        const questionResults = questions.map((q, idx) => {
          const answerLength = answers[idx].length;
          // Score based on answer completeness and live metrics
          const baseScore = answerLength > 200 ? 85 : answerLength > 100 ? 75 : answerLength > 50 ? 65 : 50;
          const metricsBonus = liveMetrics.confidence > 70 ? 10 : liveMetrics.confidence > 50 ? 5 : 0;
          const score = Math.min(100, baseScore + metricsBonus);
          
          return {
            title: q.title,
            category: q.category,
            difficulty: q.difficulty,
            question: q.question,
            answer: answers[idx],
            score,
            feedback: answerLength > 100 
              ? "Good detailed response with clear explanation."
              : "Consider providing more detail in your response."
          };
        });
        
        const technicalScore = Math.floor(questionResults.reduce((acc, q) => acc + (q.score || 0), 0) / questionResults.length);
        
        // Use real speech test results or live audio metrics
        const communicationScore = speechTestResults 
          ? Math.floor((speechTestResults.fluency + speechTestResults.pronunciation) / 2)
          : liveMetrics.clarity > 0 || liveMetrics.fluency > 0
            ? Math.floor((liveMetrics.clarity + liveMetrics.fluency) / 2)
            : 0;
        
        // Use real body language metrics from video analysis
        const bodyLanguageScore = metrics.isConnected && liveMetrics.faceDetected
          ? Math.floor((liveMetrics.eyeContact + liveMetrics.attention + liveMetrics.confidence) / 3)
          : 0;
        
        const overallScore = Math.floor((technicalScore + communicationScore + bodyLanguageScore) / 3);
        
        // Save results with timeout
        if (currentUser) {
          toast.info("Saving results to Firebase...");
          
          // Create a timeout promise that rejects after 5 seconds
          const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error("Save operation timed out")), 5000)
          );
          
          try {
            // Race between the save operation and timeout
            await Promise.race([
              completeSession({
                overallScore,
                technicalScore,
                communicationScore,
                bodyLanguageScore,
                bodyLanguage: {
                  eyeContact: liveMetrics.eyeContact,
                  avgBlinkRate: liveMetrics.blinkRate,
                  confidenceCurve: liveMetrics.confidence,
                  emotionTimeline: ["Confident", "Focused", liveMetrics.emotion]
                },
                questions: questionResults
              }),
              timeoutPromise
            ]);
            toast.success("Interview completed and saved!");
          } catch (timeoutError) {
            // If timeout or error, save locally and continue
            console.warn("Firebase save timeout or error, saving locally", timeoutError);
            toast.warning("Saved locally - Firebase sync may have issues");
            // Store in localStorage as backup
            try {
              localStorage.setItem("interview_results", JSON.stringify({
                overallScore,
                technicalScore,
                communicationScore,
                bodyLanguageScore,
                bodyLanguage: {
                  eyeContact: liveMetrics.eyeContact,
                  avgBlinkRate: liveMetrics.blinkRate,
                  confidenceCurve: liveMetrics.confidence,
                  emotionTimeline: ["Confident", "Focused", liveMetrics.emotion]
                },
                questions: questionResults
              }));
            } catch (e) {
              console.error("Failed to save to localStorage:", e);
            }
          }
        } else {
          toast.info("Saving results locally (not logged in)");
          // Save to localStorage for non-logged-in users
          try {
            localStorage.setItem("interview_results", JSON.stringify({
              overallScore,
              technicalScore,
              communicationScore,
              bodyLanguageScore,
              bodyLanguage: {
                eyeContact: liveMetrics.eyeContact,
                avgBlinkRate: liveMetrics.blinkRate,
                confidenceCurve: liveMetrics.confidence,
                emotionTimeline: ["Confident", "Focused", liveMetrics.emotion]
              },
              questions: questionResults
            }));
          } catch (e) {
            console.error("Failed to save to localStorage:", e);
          }
        }
      } catch (error) {
        console.error("Error preparing interview results:", error);
        const errorMessage = error instanceof Error ? error.message : "Unknown error";
        toast.error(`Error: ${errorMessage}`);
      } finally {
        setIsSubmitting(false);
      }
      
      // Always navigate to report, regardless of save status
      stopVideo();
      setTimeout(() => {
        navigate("/report");
      }, 500);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const MetricGauge = ({ label, value, icon: Icon, color, showProgress = true }: any) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2">
          <Icon className={`h-4 w-4 ${color}`} />
          <span className="font-medium">{label}</span>
        </div>
        <span className="font-bold">{typeof value === 'number' ? `${Math.round(value)}%` : value}</span>
      </div>
      {showProgress && typeof value === 'number' && <Progress value={value} className="h-2" />}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Button variant="ghost" onClick={() => navigate("/speech-test")} disabled>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <div className="flex items-center gap-4">
            <Badge variant="secondary">Step 2 of 2</Badge>
            {/* Connection status indicator */}
            <Badge variant={metrics.isConnected ? "default" : "destructive"}>
              {metrics.isConnected ? "🟢 Connected" : "🔴 Disconnected"}
            </Badge>
            <div className="flex items-center gap-2 text-sm font-medium">
              <Clock className="h-4 w-4" />
              {formatTime(timeRemaining)}
            </div>
          </div>
          <div className="w-20" />
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Sidebar - Question Navigation */}
          <div className="col-span-2 space-y-3">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Questions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {questions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => setCurrentQuestion(idx)}
                    className={`w-full text-left p-3 rounded-lg text-sm transition-colors ${
                      currentQuestion === idx
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary hover:bg-secondary/80"
                    }`}
                  >
                    <div className="font-medium">Q{idx + 1}</div>
                    <div className="text-xs opacity-80">{q.title}</div>
                  </button>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Center - Main Interview Area */}
          <div className="col-span-7 space-y-6">
            <Card className="border-accent/20">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-2xl font-heading">
                      {questions[currentQuestion].title}
                    </CardTitle>
                    <CardDescription className="mt-2">
                      {questions[currentQuestion].category}
                    </CardDescription>
                  </div>
                  <Badge variant={
                    questions[currentQuestion].difficulty === "Hard" ? "destructive" :
                    questions[currentQuestion].difficulty === "Medium" ? "default" : "secondary"
                  }>
                    {questions[currentQuestion].difficulty}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                <p className="text-lg leading-relaxed">
                  {questions[currentQuestion].question}
                </p>

                <Textarea
                  placeholder="Type your answer here or click the microphone to speak..."
                  value={answers[currentQuestion]}
                  onChange={(e) => handleAnswerChange(e.target.value)}
                  className="min-h-[200px] text-base"
                />

                {/* Transcribed text from speech */}
                {liveMetrics.transcribedText && (
                  <div className="p-3 bg-secondary rounded-lg">
                    <p className="text-xs text-muted-foreground mb-1">Transcribed Speech:</p>
                    <p className="text-sm">{liveMetrics.transcribedText}</p>
                  </div>
                )}

                <div className="flex gap-3">
                  <Button 
                    variant={isRecording ? "destructive" : "outline"}
                    size="lg"
                    onClick={toggleRecording}
                  >
                    {isRecording ? (
                      <>
                        <MicOff className="mr-2 h-4 w-4" />
                        Stop Recording
                      </>
                    ) : (
                      <>
                        <Mic className="mr-2 h-4 w-4" />
                        Record Answer
                      </>
                    )}
                  </Button>
                  <Button onClick={handleNextQuestion} size="lg" disabled={isSubmitting}>
                    {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    {currentQuestion < questions.length - 1 ? "Next Question" : "Finish Interview"}
                  </Button>
                  <Button variant="outline" size="lg">
                    Hint
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Camera Feed Preview */}
            <Card>
              <CardContent className="pt-6">
                <div className="aspect-video bg-muted rounded-lg relative overflow-hidden">
                  {/* Actual video element */}
                  <video
                    ref={videoElementRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover scale-x-[-1]"
                  />
                  
                  {/* Overlay indicators */}
                  <div className="absolute top-4 left-4 space-y-2">
                    <Badge className={liveMetrics.isLooking ? "bg-success" : "bg-destructive"}>
                      Eye Contact: {Math.round(liveMetrics.eyeContact)}%
                    </Badge>
                    <Badge className="bg-accent">{liveMetrics.emotion}</Badge>
                    {!liveMetrics.faceDetected && (
                      <Badge className="bg-warning flex items-center gap-1">
                        <AlertCircle className="h-3 w-3" />
                        No face detected
                      </Badge>
                    )}
                  </div>

                  {/* Recording indicator */}
                  {isRecording && (
                    <div className="absolute top-4 right-4">
                      <Badge variant="destructive" className="animate-pulse flex items-center gap-1">
                        <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                        Recording
                      </Badge>
                    </div>
                  )}

                  {/* Head pose indicator */}
                  <div className="absolute bottom-4 left-4 text-xs text-white bg-black/50 px-2 py-1 rounded">
                    Yaw: {liveMetrics.yaw.toFixed(1)}° | Pitch: {liveMetrics.pitch.toFixed(1)}° | Roll: {liveMetrics.roll.toFixed(1)}°
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Sidebar - Live Metrics */}
          <div className="col-span-3 space-y-4">
            <Card className="sticky top-24">
              <CardHeader>
                <CardTitle className="text-base">📊 Live Metrics</CardTitle>
                <CardDescription className="text-xs">Real-time performance</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <MetricGauge
                  label="Attention"
                  value={liveMetrics.attention}
                  icon={Eye}
                  color="text-accent"
                />
                <MetricGauge
                  label="Eye Contact"
                  value={liveMetrics.eyeContact}
                  icon={Eye}
                  color="text-success"
                />
                <MetricGauge
                  label="Confidence"
                  value={liveMetrics.confidence}
                  icon={Activity}
                  color="text-primary"
                />
                <MetricGauge
                  label="Emotion"
                  value={liveMetrics.emotion}
                  icon={Smile}
                  color="text-warning"
                  showProgress={false}
                />

                {/* Speech Metrics */}
                {(liveMetrics.clarity > 0 || liveMetrics.fluency > 0) && (
                  <div className="pt-4 border-t space-y-3">
                    <p className="text-xs font-medium text-muted-foreground">🎤 Speech Analysis</p>
                    <MetricGauge
                      label="Clarity"
                      value={liveMetrics.clarity}
                      icon={Mic}
                      color="text-blue-500"
                    />
                    <MetricGauge
                      label="Fluency"
                      value={liveMetrics.fluency}
                      icon={Activity}
                      color="text-purple-500"
                    />
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">Speaking Pace</span>
                      <span className="font-bold">{Math.round(liveMetrics.pace)} WPM</span>
                    </div>
                    {liveMetrics.fillerWords > 0 && (
                      <div className="flex items-center justify-between text-sm text-warning">
                        <span className="font-medium">Filler Words</span>
                        <span className="font-bold">{liveMetrics.fillerWords}</span>
                      </div>
                    )}
                  </div>
                )}

                <div className="pt-4 border-t space-y-3">
                  <div className="text-xs space-y-1">
                    <p className="font-medium">🎯 Quick Tips</p>
                    {!liveMetrics.isLooking && liveMetrics.faceDetected && (
                      <p className="text-muted-foreground">⚠️ Look at the camera to improve eye contact</p>
                    )}
                    {!liveMetrics.faceDetected && (
                      <p className="text-muted-foreground">⚠️ Please position your face in the camera view</p>
                    )}
                    {liveMetrics.isLooking && (
                      <p className="text-muted-foreground">✅ Great eye contact! Keep it up</p>
                    )}
                    {liveMetrics.pace > 180 && (
                      <p className="text-muted-foreground">⚠️ You're speaking fast - try to slow down</p>
                    )}
                    {liveMetrics.pace > 0 && liveMetrics.pace < 100 && (
                      <p className="text-muted-foreground">⚠️ You're speaking slowly - try to be more fluent</p>
                    )}
                  </div>
                  
                  {/* Status indicators */}
                  <div className="h-20 bg-muted rounded flex flex-col items-center justify-center gap-1">
                    <div className="flex items-center gap-2 text-xs">
                      {metrics.isVideoActive ? (
                        <Video className="h-4 w-4 text-success" />
                      ) : (
                        <VideoOff className="h-4 w-4 text-destructive" />
                      )}
                      <span>{metrics.isVideoActive ? "Camera Active" : "Camera Off"}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      {isRecording ? (
                        <Mic className="h-4 w-4 text-success animate-pulse" />
                      ) : (
                        <MicOff className="h-4 w-4 text-muted-foreground" />
                      )}
                      <span>{isRecording ? "Recording..." : "Microphone Ready"}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Interview;
