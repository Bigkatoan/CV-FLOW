import { create } from 'zustand'

interface UIState {
  leftPanelOpen: boolean
  rightPanelOpen: boolean
  modelHubOpen: boolean
  executionPanelOpen: boolean
  activeRightTab: 'properties' | 'logs'

  toggleLeftPanel: () => void
  toggleRightPanel: () => void
  toggleModelHub: () => void
  toggleExecutionPanel: () => void
  setActiveRightTab: (tab: 'properties' | 'logs') => void
}

export const useUIStore = create<UIState>((set) => ({
  leftPanelOpen: true,
  rightPanelOpen: true,
  modelHubOpen: false,
  executionPanelOpen: true,
  activeRightTab: 'properties',

  toggleLeftPanel: () => set((s) => ({ leftPanelOpen: !s.leftPanelOpen })),
  toggleRightPanel: () => set((s) => ({ rightPanelOpen: !s.rightPanelOpen })),
  toggleModelHub: () => set((s) => ({ modelHubOpen: !s.modelHubOpen })),
  toggleExecutionPanel: () => set((s) => ({ executionPanelOpen: !s.executionPanelOpen })),
  setActiveRightTab: (tab) => set({ activeRightTab: tab }),
}))
