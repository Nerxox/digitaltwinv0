export interface EnergyDataPoint {
  timestamp: string;
  actual: number;
  predicted?: number;
  building_id: string;
}

export interface Building {
  id: string;
  name: string;
  location: string;
  lastUpdated: string;
  energyUsage: number;
  status: 'online' | 'offline' | 'warning';
}

export interface TimeRange {
  label: string;
  value: string;
  days: number;
}
