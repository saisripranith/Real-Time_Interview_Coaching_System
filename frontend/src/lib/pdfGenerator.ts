import jsPDF from "jspdf";
import type { InterviewSession } from "./firestore";

export const generateInterviewReport = (session: InterviewSession, userName?: string): void => {
  const doc = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  let yPos = 20;
  
  const addText = (text: string, x: number, y: number, options?: { fontSize?: number; fontStyle?: string; color?: [number, number, number] }) => {
    if (options?.fontSize) doc.setFontSize(options.fontSize);
    if (options?.fontStyle) doc.setFont("helvetica", options.fontStyle);
    if (options?.color) doc.setTextColor(...options.color);
    else doc.setTextColor(0, 0, 0);
    doc.text(text, x, y);
    return y;
  };

  const addSection = (title: string) => {
    yPos += 10;
    doc.setDrawColor(59, 130, 246);
    doc.setLineWidth(0.5);
    doc.line(15, yPos, pageWidth - 15, yPos);
    yPos += 8;
    addText(title, 15, yPos, { fontSize: 14, fontStyle: "bold", color: [59, 130, 246] });
    yPos += 8;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(11);
  };

  // Header
  doc.setFillColor(59, 130, 246);
  doc.rect(0, 0, pageWidth, 40, "F");
  addText("Interview Assessment Report", pageWidth / 2 - 45, 20, { fontSize: 20, fontStyle: "bold", color: [255, 255, 255] });
  addText("Interview Coach - Performance Analysis", pageWidth / 2 - 40, 30, { fontSize: 11, fontStyle: "normal", color: [255, 255, 255] });
  
  yPos = 55;
  
  // User info and date
  if (userName) {
    addText(`Candidate: ${userName}`, 15, yPos, { fontSize: 11, fontStyle: "normal" });
    yPos += 6;
  }
  addText(`Date: ${new Date().toLocaleDateString("en-US", { 
    weekday: "long", 
    year: "numeric", 
    month: "long", 
    day: "numeric" 
  })}`, 15, yPos, { fontSize: 11 });
  
  // Overall Score Section
  addSection("Overall Performance");
  
  const overallScore = session.overallScore || 0;
  const scoreColor: [number, number, number] = overallScore >= 80 ? [34, 197, 94] : overallScore >= 60 ? [251, 191, 36] : [239, 68, 68];
  
  addText(`Overall Score: ${overallScore}/100`, 15, yPos, { fontSize: 16, fontStyle: "bold", color: scoreColor });
  yPos += 8;
  
  const getPerformanceLevel = (score: number) => {
    if (score >= 90) return "Excellent";
    if (score >= 80) return "Above Average";
    if (score >= 70) return "Good";
    if (score >= 60) return "Average";
    return "Needs Improvement";
  };
  
  addText(`Performance Level: ${getPerformanceLevel(overallScore)}`, 15, yPos, { fontSize: 11 });
  yPos += 12;
  
  // Score Breakdown
  const scores = [
    { label: "Technical Score", value: session.technicalScore || 0 },
    { label: "Communication Score", value: session.communicationScore || 0 },
    { label: "Body Language Score", value: session.bodyLanguageScore || 0 }
  ];
  
  scores.forEach(score => {
    addText(`${score.label}: ${score.value}/100`, 20, yPos, { fontSize: 11 });
    yPos += 6;
  });
  
  // Speech Analysis Section
  if (session.speechTest) {
    addSection("Speech Analysis");
    
    addText(`Fluency Score: ${session.speechTest.fluency}/100`, 20, yPos, { fontSize: 11 });
    yPos += 6;
    addText(`Pronunciation: ${session.speechTest.pronunciation}/100`, 20, yPos, { fontSize: 11 });
    yPos += 6;
    addText(`Filler Words Detected: ${session.speechTest.fillerWords}`, 20, yPos, { fontSize: 11 });
    yPos += 6;
    addText(`Speaking Pace: ${session.speechTest.pace} WPM`, 20, yPos, { fontSize: 11 });
    yPos += 6;
    addText(`Recording Duration: ${session.speechTest.recordingDuration} seconds`, 20, yPos, { fontSize: 11 });
  }
  
  // Body Language Section
  if (session.bodyLanguage) {
    addSection("Body Language Analysis");
    
    addText(`Eye Contact: ${session.bodyLanguage.eyeContact}%`, 20, yPos, { fontSize: 11 });
    yPos += 6;
    addText(`Average Blink Rate: ${session.bodyLanguage.avgBlinkRate}/min`, 20, yPos, { fontSize: 11 });
    yPos += 6;
    addText(`Confidence Score: ${session.bodyLanguage.confidenceCurve}%`, 20, yPos, { fontSize: 11 });
    yPos += 6;
    
    if (session.bodyLanguage.emotionTimeline.length > 0) {
      addText(`Detected Emotions: ${session.bodyLanguage.emotionTimeline.join(", ")}`, 20, yPos, { fontSize: 11 });
    }
  }
  
  // Questions and Answers Section
  if (session.questions && session.questions.length > 0) {
    addSection("Technical Questions Review");
    
    session.questions.forEach((q, idx) => {
      // Check if we need a new page
      if (yPos > 250) {
        doc.addPage();
        yPos = 20;
      }
      
      addText(`Question ${idx + 1}: ${q.title}`, 15, yPos, { fontSize: 12, fontStyle: "bold" });
      yPos += 6;
      addText(`Category: ${q.category} | Difficulty: ${q.difficulty}`, 20, yPos, { fontSize: 10, color: [107, 114, 128] });
      yPos += 6;
      
      if (q.score !== undefined) {
        const qScoreColor: [number, number, number] = q.score >= 80 ? [34, 197, 94] : q.score >= 60 ? [251, 191, 36] : [239, 68, 68];
        addText(`Score: ${q.score}/100`, 20, yPos, { fontSize: 11, color: qScoreColor });
        yPos += 6;
      }
      
      if (q.feedback) {
        const feedbackLines = doc.splitTextToSize(`Feedback: ${q.feedback}`, pageWidth - 40);
        feedbackLines.forEach((line: string) => {
          addText(line, 20, yPos, { fontSize: 10 });
          yPos += 5;
        });
      }
      
      yPos += 8;
    });
  }
  
  // Footer
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(9);
    doc.setTextColor(128, 128, 128);
    doc.text(`Page ${i} of ${pageCount}`, pageWidth / 2, doc.internal.pageSize.getHeight() - 10, { align: "center" });
    doc.text("Generated by Interview Coach", 15, doc.internal.pageSize.getHeight() - 10);
  }
  
  // Save the PDF
  const fileName = `interview-report-${new Date().toISOString().split("T")[0]}.pdf`;
  doc.save(fileName);
};
