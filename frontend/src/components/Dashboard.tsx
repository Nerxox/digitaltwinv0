import { useState, useEffect } from 'react';
import { Building, EnergyDataPoint, TimeRange } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { format, subDays } from 'date-fns';
import axios from 'axios';

const timeRanges: TimeRange[] = [
  { label: '24h', value: '24h', days: 1 },
  { label: '7d', value: '7d', days: 7 },
  { label: '30d', value: '30d', days: 30 },
  { label: '90d', value: '90d', days: 90 },
];

const mockBuildings: Building[] = [
  { id: 'M001', name: 'Main Office', location: 'Building A', lastUpdated: new Date().toISOString(), energyUsage: 1250, status: 'online' },
  { id: 'M002', name: 'Data Center', location: 'Building B', lastUpdated: new Date().toISOString(), energyUsage: 3540, status: 'online' },
  { id: 'M003', name: 'R&D Lab', location: 'Building C', lastUpdated: new Date().toISOString(), energyUsage: 890, status: 'warning' },
];

const generateMockData = (buildingId: string, days: number): EnergyDataPoint[] => {
  const data: EnergyDataPoint[] = [];
  const now = new Date();

  for (let i = 0; i < 24 * days; i++) {
    const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
    const baseValue = 500 + Math.random() * 1000;
    const predicted = baseValue * (0.95 + Math.random() * 0.1);

    data.push({
      timestamp: timestamp.toISOString(),
      actual: Math.round(baseValue * (0.9 + Math.random() * 0.2)),
      predicted: Math.round(predicted),
      building_id: buildingId,
    });
  }

  return data.reverse();
};

export default function Dashboard() {
  const [selectedBuilding, setSelectedBuilding] = useState<Building>(mockBuildings[0]);
  const [timeRange, setTimeRange] = useState<TimeRange>(timeRanges[0]);
  const [energyData, setEnergyData] = useState<EnergyDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // In a real app, you would fetch this from your FastAPI backend
        // const response = await axios.get(`/api/energy-data/${selectedBuilding.id}?days=${timeRange.days}`);
        // setEnergyData(response.data);

        // For now, using mock data
        const mockData = generateMockData(selectedBuilding.id, timeRange.days);
        setEnergyData(mockData);
      } catch (error) {
        console.error('Error fetching energy data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [selectedBuilding, timeRange]);

  const formatXAxis = (timestamp: string) => {
    return format(new Date(timestamp), timeRange.days <= 1 ? 'HH:mm' : 'MMM d');
  };

  const getTotalEnergy = () => {
    return energyData.reduce((sum, point) => sum + point.actual, 0);
  };

  const getAvgEnergy = () => {
    return energyData.length > 0 ? Math.round(getTotalEnergy() / energyData.length) : 0;
  };

  const getPeakUsage = () => {
    return energyData.length > 0 ? Math.max(...energyData.map(point => point.actual)) : 0;
  };

  const getEfficiency = () => {
    const totalPredicted = energyData.reduce((sum, point) => sum + (point.predicted || 0), 0);
    const totalActual = getTotalEnergy();
    return totalPredicted > 0 ? Math.round((totalActual / totalPredicted) * 100) : 0;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Digital Twin Dashboard</h1>
              <p className="text-sm text-gray-600 mt-1">Real-time energy monitoring and prediction</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-sm text-gray-500">Last Updated</div>
                <div className="text-sm font-medium text-gray-900">
                  {format(new Date(selectedBuilding.lastUpdated), 'MMM d, HH:mm')}
                </div>
              </div>
              <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Buildings</h2>
              <div className="space-y-2">
                {mockBuildings.map((building) => (
                  <button
                    key={building.id}
                    onClick={() => setSelectedBuilding(building)}
                    className={`w-full text-left p-3 rounded-lg transition-all ${
                      selectedBuilding.id === building.id
                        ? 'bg-blue-50 text-blue-700 border-2 border-blue-200'
                        : 'hover:bg-gray-50 border-2 border-transparent'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <div className="font-medium">{building.name}</div>
                        <div className="text-sm text-gray-500">{building.location}</div>
                      </div>
                      <span className={`inline-block w-3 h-3 rounded-full ${
                        building.status === 'online' ? 'bg-green-500' :
                        building.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                      }`}></span>
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      {building.energyUsage.toLocaleString()} kWh
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Controls */}
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">
                    {selectedBuilding.name} - Energy Consumption
                  </h2>
                  <p className="text-sm text-gray-600 mt-1">
                    Building ID: {selectedBuilding.id} • Location: {selectedBuilding.location}
                  </p>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="flex bg-gray-100 rounded-lg p-1">
                    <button
                      onClick={() => setViewMode('chart')}
                      className={`px-3 py-1 text-sm rounded-md transition-colors ${
                        viewMode === 'chart'
                          ? 'bg-white text-gray-900 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      Chart
                    </button>
                    <button
                      onClick={() => setViewMode('table')}
                      className={`px-3 py-1 text-sm rounded-md transition-colors ${
                        viewMode === 'table'
                          ? 'bg-white text-gray-900 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      Table
                    </button>
                  </div>
                  <div className="flex bg-gray-100 rounded-lg p-1">
                    {timeRanges.map((range) => (
                      <button
                        key={range.value}
                        onClick={() => setTimeRange(range)}
                        className={`px-3 py-1 text-sm rounded-md transition-colors ${
                          timeRange.value === range.value
                            ? 'bg-white text-gray-900 shadow-sm'
                            : 'text-gray-600 hover:text-gray-900'
                        }`}
                      >
                        {range.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="flex items-center">
                  <div className="p-3 rounded-full bg-blue-100 text-blue-600">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <div className="ml-4">
                    <h3 className="text-sm font-medium text-gray-500">Total Usage</h3>
                    <p className="text-2xl font-bold text-gray-900">
                      {getTotalEnergy().toLocaleString()} kWh
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="flex items-center">
                  <div className="p-3 rounded-full bg-green-100 text-green-600">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <div className="ml-4">
                    <h3 className="text-sm font-medium text-gray-500">Average</h3>
                    <p className="text-2xl font-bold text-gray-900">
                      {getAvgEnergy().toLocaleString()} kWh
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="flex items-center">
                  <div className="p-3 rounded-full bg-orange-100 text-orange-600">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                  <div className="ml-4">
                    <h3 className="text-sm font-medium text-gray-500">Peak Usage</h3>
                    <p className="text-2xl font-bold text-gray-900">
                      {getPeakUsage().toLocaleString()} kWh
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="flex items-center">
                  <div className="p-3 rounded-full bg-purple-100 text-purple-600">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="ml-4">
                    <h3 className="text-sm font-medium text-gray-500">Efficiency</h3>
                    <p className="text-2xl font-bold text-gray-900">
                      {getEfficiency()}%
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Chart/Table */}
            <div className="bg-white shadow rounded-lg p-6">
              {isLoading ? (
                <div className="flex items-center justify-center h-80">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                </div>
              ) : viewMode === 'chart' ? (
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={energyData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="timestamp"
                        tickFormatter={formatXAxis}
                        tick={{ fontSize: 12, fill: '#6b7280' }}
                        axisLine={{ stroke: '#d1d5db' }}
                      />
                      <YAxis
                        label={{ value: 'kWh', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
                        tick={{ fontSize: 12, fill: '#6b7280' }}
                        axisLine={{ stroke: '#d1d5db' }}
                      />
                      <Tooltip
                        labelFormatter={(value) => format(new Date(value), 'PPpp')}
                        formatter={(value: number) => [`${value} kWh`, 'Energy']}
                        contentStyle={{
                          backgroundColor: '#fff',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="actual"
                        name="Actual"
                        stroke="#3b82f6"
                        strokeWidth={3}
                        dot={false}
                        activeDot={{ r: 5, fill: '#3b82f6' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        name="Predicted"
                        stroke="#10b981"
                        strokeWidth={3}
                        strokeDasharray="8 8"
                        dot={false}
                        activeDot={{ r: 5, fill: '#10b981' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Time
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actual (kWh)
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Predicted (kWh)
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Variance
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {energyData.slice(0, 20).map((point, index) => (
                        <tr key={index} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {format(new Date(point.timestamp), 'MMM d, HH:mm')}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {point.actual.toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {point.predicted?.toLocaleString() || '-'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm">
                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              point.predicted && Math.abs(point.actual - point.predicted) < 50
                                ? 'bg-green-100 text-green-800'
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {point.predicted ? `${((point.actual - point.predicted) / point.predicted * 100).toFixed(1)}%` : '-'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
