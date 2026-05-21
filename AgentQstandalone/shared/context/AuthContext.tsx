
import React, { createContext, useContext, useState, ReactNode } from 'react';

export type UserLevel = 1 | 2 | 3 | 4;

interface AuthContextType {
  isAuthenticated: boolean;
  adminLevel: UserLevel | 0;
  currentUser: string | null;
  token: string | null;
  login: (username: string, level: UserLevel) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(true); // Standalone defaults to active
  const [adminLevel, setAdminLevel] = useState<UserLevel | 0>(4);
  const [currentUser, setCurrentUser] = useState<string | null>("AgentQ_PRIME");
  const [token, setToken] = useState<string | null>("mock_standalone_token");

  const login = (username: string, level: UserLevel) => {
    setIsAuthenticated(true);
    setCurrentUser(username);
    setAdminLevel(level);
    setToken("mock_standalone_token");
  };

  const logout = () => {
    setIsAuthenticated(false);
    setAdminLevel(0);
    setCurrentUser(null);
    setToken(null);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, adminLevel, currentUser, token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
