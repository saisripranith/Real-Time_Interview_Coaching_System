import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Mic, Video, Clock, Target, TrendingUp, CheckCircle2 } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Mic,
      title: "Speech Test",
      description: "Pronunciation Analysis",
      details: ["Fluency Score", "Filler-word Detection", "Pace Analysis"],
      duration: "3 minutes",
      color: "text-accent"
    },
    {
      icon: Video,
      title: "Technical Interview",
      description: "Role-based Questions",
      details: ["AI Scoring & Explanations", "Body Language Detection", "Real-time Feedback"],
      duration: "30 minutes",
      color: "text-primary"
    }
  ];

  const requirements = [
    { label: "Laptop/PC", met: true },
    { label: "Camera (720p+)", met: true },
    { label: "Microphone", met: true },
    { label: "Stable lighting", met: true }
  ];

  return (
    <div className="min-h-screen bg-gradient-subtle">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Target className="h-6 w-6 text-primary" />
            <span className="text-xl font-heading font-bold text-primary">Interview Coach</span>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost">About</Button>
            <Button variant="ghost">Help</Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 text-center">
        <Badge className="mb-4" variant="secondary">
          AI-Powered Assessment Platform
        </Badge>
        <h1 className="text-5xl md:text-6xl font-heading font-bold text-primary mb-6 leading-tight">
          Real-Time Interview<br />Coaching System
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
          Assess your speaking clarity, confidence, and technical skills with AI-powered feedback and real-time body language analysis
        </p>
        <div className="flex gap-4 justify-center">
          <Button size="lg" onClick={() => navigate("/login")} className="gap-2">
            Begin Assessment
            <TrendingUp className="h-5 w-5" />
          </Button>
          <Button size="lg" variant="outline">
            Watch Demo
          </Button>
        </div>
      </section>

      {/* Test Overview Cards */}
      <section className="container mx-auto px-4 py-12">
        <h2 className="text-3xl font-heading font-bold text-center mb-8">Two-Stage Assessment</h2>
        <div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto">
          {features.map((feature, idx) => (
            <Card key={idx} className="border-accent/20 hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`p-3 rounded-lg bg-accent/10`}>
                      <feature.icon className={`h-6 w-6 ${feature.color}`} />
                    </div>
                    <div>
                      <CardTitle className="text-xl">{feature.title}</CardTitle>
                      <CardDescription className="mt-1">{feature.description}</CardDescription>
                    </div>
                  </div>
                  <Badge variant="outline" className="gap-1">
                    <Clock className="h-3 w-3" />
                    {feature.duration}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {feature.details.map((detail, i) => (
                    <li key={i} className="flex items-center gap-2 text-sm">
                      <CheckCircle2 className="h-4 w-4 text-success" />
                      {detail}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Requirements Section */}
      <section className="container mx-auto px-4 py-12">
        <Card className="max-w-3xl mx-auto bg-gradient-primary text-primary-foreground">
          <CardHeader>
            <CardTitle className="text-2xl font-heading">System Requirements</CardTitle>
            <CardDescription className="text-primary-foreground/80">
              Ensure you have the following before starting
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              {requirements.map((req, idx) => (
                <div key={idx} className="flex items-center gap-3">
                  <CheckCircle2 className="h-5 w-5 text-success" />
                  <span className="font-medium">{req.label}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-16 text-center">
        <div className="max-w-2xl mx-auto">
          <h2 className="text-3xl font-heading font-bold mb-4">Ready to Begin?</h2>
          <p className="text-muted-foreground mb-8">
            Start your comprehensive interview assessment and get detailed feedback on your performance
          </p>
          <Button size="lg" onClick={() => navigate("/login")} className="gap-2">
            Start Test Now
            <TrendingUp className="h-5 w-5" />
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-card/50 backdrop-blur-sm py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <div className="flex justify-center gap-6 mb-4">
            <a href="#" className="hover:text-foreground transition-colors">Terms</a>
            <a href="#" className="hover:text-foreground transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-foreground transition-colors">Contact</a>
          </div>
          <p>&copy; 2024 Interview Coach. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
