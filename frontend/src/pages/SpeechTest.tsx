import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Mic, Square, ArrowLeft, ArrowRight, Loader2, ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import { useInterview } from "@/contexts/InterviewContext";
import { useAuth } from "@/contexts/AuthContext";
import { useInterviewMetrics } from "@/hooks/useInterviewMetrics";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

const SpeechTest = () => {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const { saveSpeechTest, startNewSession, sessionId } = useInterview();
  const { sendAudioForAnalysis } = useInterviewMetrics();
  
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [hasRecorded, setHasRecorded] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  
  const [metrics, setMetrics] = useState({
    fluency: 0,
    fillerWords: 0,
    pace: 0,
    pronunciation: 0,
  });
  
  const [analysisData, setAnalysisData] = useState({
    transcribedText: "",
    fillerWordsDetail: {} as Record<string, number>,
    clarityScore: 0,
  });
  
  const [isTranscriptionOpen, setIsTranscriptionOpen] = useState(false);

  const prompt = "Describe a recent project you worked on and explain your role in it. What challenges did you face and how did you overcome them?";

  const startRecording = async () => {
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(1000); // Collect data every second
      
      // Start recording UI updates
      setIsRecording(true);
      setRecordingTime(0);
      intervalRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
      
      toast.success("Recording started - speak clearly");
      
      // Create session in background if user is logged in and no session exists
      if (currentUser && !sessionId) {
        startNewSession().catch((error) => {
          console.error("Failed to create session:", error);
        });
      }
    } catch (error) {
      console.error("Failed to access microphone:", error);
      toast.error("Failed to access microphone. Please check permissions.");
    }
  };

  const stopRecording = async () => {
    setIsRecording(false);
    const finalRecordingTime = recordingTime;
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    if (!mediaRecorderRef.current) {
      return;
    }
    
    setIsAnalyzing(true);
    toast.info("Analyzing your speech...");
    
    // Stop the media recorder and wait for final data
    const audioBlob = await new Promise<Blob>((resolve) => {
      mediaRecorderRef.current!.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        resolve(blob);
      };
      mediaRecorderRef.current!.stop();
    });
    
    // Stop the audio stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    try {
      // Send audio to backend for real analysis
      const analysisResult = await sendAudioForAnalysis(audioBlob, prompt);
      
      const newMetrics = {
        fluency: Math.round(analysisResult.fluency_score || 0),
        fillerWords: analysisResult.filler_words_count || 0,
        pace: Math.round(analysisResult.pace_wpm || 0),
        pronunciation: Math.round(analysisResult.pronunciation_score || 0),
      };
      
      setMetrics(newMetrics);
      setAnalysisData({
        transcribedText: analysisResult.transcribed_text || "",
        fillerWordsDetail: analysisResult.filler_words_detail || {},
        clarityScore: Math.round(analysisResult.clarity_score || 0),
      });
      setHasRecorded(true);
      
      // Save to Firebase if user is logged in
      if (currentUser) {
        setIsSaving(true);
        try {
          await saveSpeechTest({
            ...newMetrics,
            recordingDuration: finalRecordingTime,
            transcribedText: analysisResult.transcribed_text,
            clarityScore: analysisResult.clarity_score,
            fillerWordsDetail: analysisResult.filler_words_detail || {},
          });
          toast.success("Speech test results saved!");
        } catch (error) {
          console.error("Failed to save results:", error);
          toast.error("Failed to save results, but you can continue");
        } finally {
          setIsSaving(false);
        }
      } else {
        toast.success("Speech analyzed successfully");
      }
    } catch (error) {
      console.error("Failed to analyze speech:", error);
      toast.error("Failed to analyze speech. Please check backend connection.");
      // Reset state so user can try again
      setHasRecorded(false);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const MetricCard = ({ label, value, unit, color }: { label: string; value: number; unit: string; color: string }) => (
    <Card>
      <CardContent className="pt-6">
        <div className="text-center">
          <p className="text-sm text-muted-foreground mb-2">{label}</p>
          <p className={`text-3xl font-bold ${color}`}>
            {value}
            <span className="text-lg text-muted-foreground ml-1">{unit}</span>
          </p>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Button variant="ghost" onClick={() => navigate("/login")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <div className="flex items-center gap-2">
            <Badge variant="secondary">Step 1 of 2</Badge>
          </div>
          <div className="w-20" />
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-heading font-bold text-primary mb-3">
            Speech Accuracy Assessment
          </h1>
          <p className="text-lg text-muted-foreground">
            Evaluate your pronunciation, fluency, and communication clarity
          </p>
        </div>

        <div className="space-y-6">
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                📝 Your Prompt
              </CardTitle>
              <CardDescription>Read this prompt aloud or respond naturally</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-lg leading-relaxed">{prompt}</p>
              <div className="mt-4 flex gap-2">
                <Badge variant="outline">Min duration: 20s</Badge>
                <Badge variant="outline">Speak clearly</Badge>
                <Badge variant="outline">Natural pace</Badge>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-primary text-primary-foreground">
            <CardContent className="pt-6 pb-6">
              <div className="flex flex-col items-center gap-6">
                <div className="relative">
                  <div className={`w-32 h-32 rounded-full bg-primary-foreground/10 flex items-center justify-center ${isRecording ? 'animate-pulse' : ''}`}>
                    <Mic className="h-16 w-16" />
                  </div>
                  {isRecording && (
                    <div className="absolute inset-0 rounded-full border-4 border-primary-foreground/30 animate-ping" />
                  )}
                </div>

                <div className="text-center">
                  <div className="text-5xl font-heading font-bold mb-2">
                    {formatTime(recordingTime)}
                  </div>
                  <p className="text-primary-foreground/80">
                    {isRecording ? "Recording in progress..." : isAnalyzing ? "Analyzing speech..." : hasRecorded ? "Recording complete" : "Ready to record"}
                  </p>
                </div>

                {!isRecording && !isAnalyzing ? (
                  <Button
                    size="lg"
                    variant="secondary"
                    onClick={startRecording}
                    className="gap-2"
                  >
                    <Mic className="h-5 w-5" />
                    Start Recording
                  </Button>
                ) : isAnalyzing ? (
                  <Button
                    size="lg"
                    variant="secondary"
                    disabled
                    className="gap-2"
                  >
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Analyzing...
                  </Button>
                ) : (
                  <Button
                    size="lg"
                    variant="destructive"
                    onClick={stopRecording}
                    className="gap-2"
                  >
                    <Square className="h-5 w-5" />
                    Stop Recording
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {hasRecorded && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>📊 Speech Analysis Results</CardTitle>
                  <CardDescription>Detailed breakdown of your speech performance</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Filler Words Section - Featured */}
                  <div className="bg-gradient-to-br from-warning/10 to-warning/5 border border-warning/30 rounded-lg p-6">
                    <div className="space-y-4">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-warning mb-1">⚠️ Filler Words Detected</h3>
                          <p className="text-sm text-muted-foreground">Words that reduce clarity and impact</p>
                        </div>
                        <span className="text-4xl font-bold text-warning">{metrics.fillerWords}</span>
                      </div>
                      
                      {/* Filler Words Detail */}
                      {Object.keys(analysisData.fillerWordsDetail).length > 0 && (
                        <div className="mt-4 pt-4 border-t border-warning/20">
                          <p className="text-sm font-medium mb-3 text-muted-foreground">Breakdown by type:</p>
                          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                            {Object.entries(analysisData.fillerWordsDetail).map(([word, count]) => (
                              <Badge key={word} variant="secondary" className="justify-center py-2 bg-warning/20 text-warning hover:bg-warning/30">
                                <span className="capitalize">{word}</span>
                                <span className="ml-1 font-bold">×{count}</span>
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Tip based on count */}
                      <div className="bg-white/50 rounded p-3 mt-4">
                        <p className="text-sm">
                          <strong>💡 Tip:</strong> {metrics.fillerWords > 5 
                            ? "Try to reduce filler words like 'um' and 'uh'. Take brief pauses instead."
                            : "Great job! Your speech is clear with minimal filler words."}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Main Metrics Grid */}
                  <div>
                    <h3 className="text-sm font-semibold mb-4 text-muted-foreground">Performance Metrics</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <MetricCard
                        label="Fluency Score"
                        value={metrics.fluency}
                        unit="/100"
                        color="text-accent"
                      />
                      <MetricCard
                        label="Speaking Pace"
                        value={metrics.pace}
                        unit="WPM"
                        color="text-blue-500"
                      />
                      <MetricCard
                        label="Pronunciation"
                        value={metrics.pronunciation}
                        unit="/100"
                        color="text-success"
                      />
                    </div>
                  </div>

                  {/* Transcription Dropdown */}
                  <Collapsible open={isTranscriptionOpen} onOpenChange={setIsTranscriptionOpen}>
                    <CollapsibleTrigger asChild>
                      <Button
                        variant="outline"
                        className="w-full justify-between"
                      >
                        <span className="font-medium">📝 View Transcription</span>
                        {isTranscriptionOpen ? (
                          <ChevronUp className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <Card className="mt-4 bg-muted/50">
                        <CardContent className="pt-4">
                          <p className="text-sm leading-relaxed text-foreground">
                            {analysisData.transcribedText}
                          </p>
                        </CardContent>
                      </Card>
                    </CollapsibleContent>
                  </Collapsible>
                </CardContent>
              </Card>

              <Button
                size="lg"
                className="w-full"
                onClick={() => navigate("/interview")}
              >
                Continue to Interview Section
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpeechTest;
