'use client';

import { createContext, useContext, useState, ReactNode } from 'react';
import { AnalysisResult } from './schemas';

interface ProfileData {
  fullName: string;
  jobTitle: string;
  roleDepartment: string;
  photoUrl: string;
}

interface AppState {
  profile: ProfileData;
  setProfile: (data: Partial<ProfileData>) => void;
  analysisResult: AnalysisResult | null;
  setAnalysisResult: (result: AnalysisResult | null) => void;
  uploadedVideo: File | null;
  uploadedImage: File | null;
  setUploadedFiles: (video: File | null, image: File | null) => void;
}

const defaultProfile: ProfileData = {
  fullName: 'Jonathan H. Sterling, Esq.',
  jobTitle: 'Senior Partner',
  roleDepartment: 'Corporate Law & Litigation',
  photoUrl: '',
};

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [profile, setProfileState] = useState<ProfileData>(defaultProfile);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);

  const setProfile = (data: Partial<ProfileData>) => {
    setProfileState((prev) => ({ ...prev, ...data }));
  };

  const setUploadedFiles = (video: File | null, image: File | null) => {
    setUploadedVideo(video);
    setUploadedImage(image);
  };

  return (
    <AppContext.Provider
      value={{
        profile,
        setProfile,
        analysisResult,
        setAnalysisResult,
        uploadedVideo,
        uploadedImage,
        setUploadedFiles,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppContext must be used within AppProvider');
  return ctx;
}
