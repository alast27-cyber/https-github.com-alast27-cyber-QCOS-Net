
import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';
import { userService, UserLevel } from '../services/userService';

// Define the shape of our authentication context
interface AuthContextType {
  isAuthenticated: boolean;
  adminLevel: UserLevel | 0; // 0 for unauthenticated
  currentUser: string | null;
  token: string | null;
  login: (username: string, level: UserLevel) => void;
  logout: () => void;
}

// Create the context with default (unauthenticated) values
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// AuthProvider component to wrap your application
interface AuthProviderProps {
  children: ReactNode;
}

// Helper: Simulate Backend JWT Generation
const createMockJwt = (payload: object) => {
  const header = { alg: "HS256", typ: "JWT" };
  const encodedHeader = btoa(JSON.stringify(header));
  
  // Add expiration (24 hours from now)
  const fullPayload = { 
    ...payload, 
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + (60 * 60 * 24) 
  };
  const encodedPayload = btoa(JSON.stringify(fullPayload));
  
  // Mock signature
  const signature = btoa(`qcos_secret_signature_${Date.now()}_${Math.random()}`);
  
  return `${encodedHeader}.${encodedPayload}.${signature}`;
};

// Helper: Parse JWT on Client Side
const parseJwt = (token: string) => {
  try {
    const base64Url = token.split('.')[1];
    if (!base64Url) return null;
    
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(window.atob(base64).split('').map(function(c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));

    return JSON.parse(jsonPayload);
  } catch (e) {
    console.error("Failed to parse JWT", e);
    return null;
  }
};

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [adminLevel, setAdminLevel] = useState<UserLevel | 0>(0);
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);

  const logout = () => {
    localStorage.removeItem('qcos_auth_token');
    
    setIsAuthenticated(false);
    setAdminLevel(0);
    setCurrentUser(null);
    setToken(null);
  };

  // Check for an existing session in localStorage on initial load
  useEffect(() => {
    const storedToken = localStorage.getItem('qcos_auth_token');

    if (storedToken) {
      const decoded = parseJwt(storedToken);

      if (decoded) {
        // Check Expiration
        const currentTime = Math.floor(Date.now() / 1000);
        if (decoded.exp && decoded.exp < currentTime) {
          console.warn("Session expired. Logging out.");
          logout();
          return;
        }

        // Validate payload structure and verify user exists in DB
        if (decoded.username && decoded.level) {
           const parsedLevel = parseInt(decoded.level, 10);
           const validLevels: Array<UserLevel> = [1, 2, 3, 4];

           // Verify against "backend" to ensure user wasn't deleted or demoted
           const dbUser = userService.getUser(decoded.username);

           if (dbUser && !isNaN(parsedLevel) && validLevels.includes(parsedLevel as UserLevel)) {
             // Optional: Force level sync with DB if it changed
             const actualLevel = dbUser.level; 
             
             setIsAuthenticated(true);
             setCurrentUser(dbUser.username);
             setAdminLevel(actualLevel);
             setToken(storedToken);
           } else {
             // User not found in DB or invalid level
             console.warn("User invalid or not found in registry.");
             logout();
           }
        } else {
          logout();
        }
      } else {
        // Invalid token format
        logout();
      }
    }
  }, []);

  const login = (username: string, level: UserLevel) => {
    // Generate a signed JWT-like token
    const newToken = createMockJwt({ username, level });
    
    localStorage.setItem('qcos_auth_token', newToken);

    setIsAuthenticated(true);
    setCurrentUser(username);
    setAdminLevel(level);
    setToken(newToken);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, adminLevel, currentUser, token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use the authentication context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
