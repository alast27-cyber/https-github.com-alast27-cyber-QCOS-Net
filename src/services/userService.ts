
import { AppDefinition } from '../types';

export type UserLevel = 1 | 2 | 3 | 4;

export interface User {
    id: string;
    username: string;
    password: string; // In a real app, this would be hashed
    level: UserLevel;
    role: string;
    lastActive?: string;
    status: 'Active' | 'Locked' | 'Flagged';
}

const STORAGE_KEY = 'qcos_users_db';

const DEFAULT_USERS: User[] = [
    { id: 'usr_admin', username: 'admin', password: 'qcos@123', level: 4, role: 'System Architect', status: 'Active' },
    { id: 'usr_redmelon', username: 'redmelon', password: 'Alexandra@123', level: 4, role: 'System Architect', status: 'Active' },
    { id: 'usr_agentq', username: 'agentq', password: 'sentience', level: 4, role: 'AGI Core', status: 'Active' },
    { id: 'usr_dev', username: 'dev', password: 'dev', level: 3, role: 'Network Admin', status: 'Active' },
    { id: 'usr_op', username: 'op', password: 'op', level: 2, role: 'Operator', status: 'Active' },
    { id: 'usr_guest', username: 'guest', password: 'guest', level: 1, role: 'Guest', status: 'Active' },
];

class UserService {
    constructor() {
        this.init();
    }

    private init() {
        if (!localStorage.getItem(STORAGE_KEY)) {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_USERS));
        }
    }

    private getUsers(): User[] {
        const stored = localStorage.getItem(STORAGE_KEY);
        return stored ? JSON.parse(stored) : [];
    }

    private saveUsers(users: User[]) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(users));
    }

    public authenticate(username: string, password: string): User | null {
        const users = this.getUsers();
        // Case-insensitive username match
        const user = users.find(u => u.username.toLowerCase() === username.toLowerCase());
        
        if (user && user.password === password) {
            if (user.status === 'Locked') {
                throw new Error('Account is locked.');
            }
            // Update last active
            user.lastActive = new Date().toISOString();
            this.updateUser(user);
            return user;
        }
        return null;
    }

    public getUser(username: string): User | undefined {
        const users = this.getUsers();
        return users.find(u => u.username.toLowerCase() === username.toLowerCase());
    }

    public getAllUsers(): User[] {
        return this.getUsers();
    }

    public addUser(user: Omit<User, 'id'>): User {
        const users = this.getUsers();
        if (users.some(u => u.username.toLowerCase() === user.username.toLowerCase())) {
            throw new Error('Username already exists');
        }
        const newUser: User = {
            ...user,
            id: `usr_${Date.now()}_${Math.floor(Math.random() * 1000)}`
        };
        users.push(newUser);
        this.saveUsers(users);
        return newUser;
    }

    public updateUser(updatedUser: User) {
        let users = this.getUsers();
        const index = users.findIndex(u => u.id === updatedUser.id);
        if (index !== -1) {
            users[index] = updatedUser;
            this.saveUsers(users);
        }
    }

    public deleteUser(userId: string) {
        let users = this.getUsers();
        users = users.filter(u => u.id !== userId);
        this.saveUsers(users);
    }
}

export const userService = new UserService();
