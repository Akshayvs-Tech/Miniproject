import { z } from 'zod';

// Profile schema
export const profileSchema = z.object({
  fullName: z.string().min(2, 'Name must be at least 2 characters').max(100),
  jobTitle: z.string().min(2, 'Job title is required').max(100),
  roleDepartment: z.string().min(2, 'Role/Department is required').max(100),
});

export type ProfileFormData = z.infer<typeof profileSchema>;

// Signup schema (no Bar Association ID)
export const signupSchema = z.object({
  fullName: z.string().min(2, 'Full name is required'),
  workEmail: z.string().email('Enter a valid work email'),
  phoneNumber: z.string().min(7, 'Enter a valid phone number'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
  agreed: z.literal(true, { errorMap: () => ({ message: 'You must agree to terms' }) }),
});

export type SignupFormData = z.infer<typeof signupSchema>;

// Login schema
export const loginSchema = z.object({
  email: z.string().email('Enter a valid email'),
  phoneNumber: z.string().min(7, 'Enter a valid phone number'),
  password: z.string().min(1, 'Password is required'),
  remember: z.boolean().optional(),
});

export type LoginFormData = z.infer<typeof loginSchema>;

// Upload schema
export const uploadSchema = z.object({
  video: z.instanceof(File, { message: 'Video file is required' }),
  image: z.instanceof(File, { message: 'Image file is required' }),
});

export type UploadFormData = z.infer<typeof uploadSchema>;

// Analysis result types
export interface Finding {
  id: string;
  title: string;
  description: string;
  tag: string;
  sourceType: 'video' | 'image';
}

export interface AnalysisResult {
  caseTitle: string;
  caseRef: string;
  generatedDate: string;
  chainOfCustody: string;
  executiveSummary: string;
  findings: Finding[];
  actionItems: string[];
}
