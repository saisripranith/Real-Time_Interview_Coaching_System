import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Camera, Mic, Clipboard, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";

const Login = () => {
  const navigate = useNavigate();
  const { currentUser, login, register, loginWithGoogle } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  
  const [permissions, setPermissions] = useState({
    camera: "pending",
    microphone: "pending",
    clipboard: false,
  });
  const [testing, setTesting] = useState({ camera: false, microphone: false });

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      toast.error("Please enter email and password");
      return;
    }
    
    setIsLoading(true);
    try {
      await login(email, password);
      toast.success("Login successful!");
    } catch (error: any) {
      toast.error(error.message || "Failed to login");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password || !confirmPassword) {
      toast.error("Please fill in all fields");
      return;
    }
    if (password !== confirmPassword) {
      toast.error("Passwords do not match");
      return;
    }
    
    setIsLoading(true);
    try {
      await register(email, password);
      toast.success("Account created successfully!");
    } catch (error: any) {
      toast.error(error.message || "Failed to create account");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setIsLoading(true);
    try {
      await loginWithGoogle();
      toast.success("Login successful!");
    } catch (error: any) {
      toast.error(error.message || "Failed to login with Google");
    } finally {
      setIsLoading(false);
    }
  };

  const testCamera = async () => {
    setTesting({ ...testing, camera: true });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setPermissions({ ...permissions, camera: "granted" });
      toast.success("Camera access granted");
      stream.getTracks().forEach(track => track.stop());
    } catch (error) {
      setPermissions({ ...permissions, camera: "denied" });
      toast.error("Camera access denied");
    }
    setTesting({ ...testing, camera: false });
  };

  const testMicrophone = async () => {
    setTesting({ ...testing, microphone: true });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setPermissions({ ...permissions, microphone: "granted" });
      toast.success("Microphone access granted");
      stream.getTracks().forEach(track => track.stop());
    } catch (error) {
      setPermissions({ ...permissions, microphone: "denied" });
      toast.error("Microphone access denied");
    }
    setTesting({ ...testing, microphone: false });
  };

  const canProceed = permissions.camera === "granted" && permissions.microphone === "granted";

  const PermissionCard = ({ 
    icon: Icon, 
    title, 
    status, 
    onTest, 
    testing 
  }: { 
    icon: any; 
    title: string; 
    status: string; 
    onTest: () => void; 
    testing: boolean;
  }) => (
    <Card>
      <CardContent className="flex items-center justify-between p-4">
        <div className="flex items-center gap-3">
          <Icon className="h-5 w-5 text-primary" />
          <span className="font-medium">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          {status === "granted" && <CheckCircle2 className="h-5 w-5 text-success" />}
          {status === "denied" && <XCircle className="h-5 w-5 text-destructive" />}
          {status === "pending" && (
            <Button onClick={onTest} size="sm" disabled={testing}>
              {testing ? <Loader2 className="h-4 w-4 animate-spin" /> : "Test"}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gradient-subtle flex items-center justify-center p-4">
      <div className="w-full max-w-2xl space-y-6">
        <div className="text-center">
          <h1 className="text-3xl font-heading font-bold text-primary mb-2">
            Interview Coach
          </h1>
          <p className="text-muted-foreground">Setup your assessment environment</p>
        </div>

        {!currentUser ? (
          <Card>
            <CardHeader>
              <CardTitle>Welcome Back</CardTitle>
              <CardDescription>Sign in to start your assessment</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="login" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="login">Login</TabsTrigger>
                  <TabsTrigger value="register">Register</TabsTrigger>
                </TabsList>
                <TabsContent value="login" className="space-y-4">
                  <form onSubmit={handleLogin} className="space-y-4">
                    <div>
                      <Input
                        type="email"
                        placeholder="Email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                      />
                    </div>
                    <div>
                      <Input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                      />
                    </div>
                    <Button type="submit" className="w-full" disabled={isLoading}>
                      {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                      Continue
                    </Button>
                  </form>
                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <span className="w-full border-t" />
                    </div>
                    <div className="relative flex justify-center text-xs uppercase">
                      <span className="bg-background px-2 text-muted-foreground">Or continue with</span>
                    </div>
                  </div>
                  <Button variant="outline" className="w-full" onClick={handleGoogleLogin} disabled={isLoading}>
                    <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
                      <path
                        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                        fill="#4285F4"
                      />
                      <path
                        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                        fill="#34A853"
                      />
                      <path
                        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                        fill="#FBBC05"
                      />
                      <path
                        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                        fill="#EA4335"
                      />
                    </svg>
                    Google
                  </Button>
                </TabsContent>
                <TabsContent value="register" className="space-y-4">
                  <form onSubmit={handleRegister} className="space-y-4">
                    <Input 
                      type="email" 
                      placeholder="Email" 
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                    />
                    <Input 
                      type="password" 
                      placeholder="Password" 
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                    />
                    <Input 
                      type="password" 
                      placeholder="Confirm Password" 
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                    />
                    <Button type="submit" className="w-full" disabled={isLoading}>
                      {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                      Create Account
                    </Button>
                  </form>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card>
              <CardHeader>
                <CardTitle>System Permissions</CardTitle>
                <CardDescription>
                  Grant access to camera and microphone for the assessment
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <PermissionCard
                  icon={Camera}
                  title="Camera Access"
                  status={permissions.camera}
                  onTest={testCamera}
                  testing={testing.camera}
                />
                <PermissionCard
                  icon={Mic}
                  title="Microphone Access"
                  status={permissions.microphone}
                  onTest={testMicrophone}
                  testing={testing.microphone}
                />
                <Card>
                  <CardContent className="flex items-center justify-between p-4">
                    <div className="flex items-center gap-3">
                      <Clipboard className="h-5 w-5 text-primary" />
                      <div>
                        <p className="font-medium">Clipboard Monitoring</p>
                        <p className="text-xs text-muted-foreground">Optional - for integrity checks</p>
                      </div>
                    </div>
                    <Button
                      variant={permissions.clipboard ? "default" : "outline"}
                      size="sm"
                      onClick={() => setPermissions({ ...permissions, clipboard: !permissions.clipboard })}
                    >
                      {permissions.clipboard ? "Enabled" : "Enable"}
                    </Button>
                  </CardContent>
                </Card>
              </CardContent>
            </Card>

            <Button
              className="w-full"
              size="lg"
              disabled={!canProceed}
              onClick={() => navigate("/speech-test")}
            >
              Continue to Speech Test
            </Button>
          </>
        )}
      </div>
    </div>
  );
};

export default Login;
