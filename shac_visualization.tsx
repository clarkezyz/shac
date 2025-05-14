import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Mock implementation of the SphericalHarmonicAudioCodec for demo purposes
const mockSHAC = {
  // Simplified spherical harmonic calculation for visualization
  sphericalHarmonic: (l, m, azimuth, elevation) => {
    // This is a simplified approximation just for visualization
    // Real implementation would use the proper formulas
    if (l === 0) {
      return 0.5;
    } else if (l === 1) {
      if (m === -1) {
        return 0.5 * Math.sin(azimuth) * Math.cos(elevation);
      } else if (m === 0) {
        return 0.5 * Math.sin(elevation);
      } else {
        return 0.5 * Math.cos(azimuth) * Math.cos(elevation);
      }
    } else {
      // For higher orders, just create some variation for visualization
      return 0.3 * Math.sin(l * azimuth + m * elevation);
    }
  },
  
  // Generate synthetic audio data for visualization
  generateAudioData: (frequency, duration, sampleRate) => {
    const numSamples = Math.floor(duration * sampleRate);
    const audioData = new Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      audioData[i] = Math.sin(2 * Math.PI * frequency * t) * Math.exp(-t / duration);
    }
    
    return audioData;
  },
  
  // Encode a mono source to ambisonics
  encodeMonoSource: (audio, position, order) => {
    const [azimuth, elevation, distance] = position;
    const numChannels = (order + 1) * (order + 1);
    const distanceGain = 1.0 / Math.max(1.0, distance);
    
    const channels = [];
    for (let i = 0; i < numChannels; i++) {
      const l = Math.floor(Math.sqrt(i));
      const m = i - l * l - l;
      
      const shVal = mockSHAC.sphericalHarmonic(l, m, azimuth, elevation);
      
      // Create channel data (simplified for visualization)
      channels.push({
        channel: i,
        degree: l,
        order: m,
        coefficient: shVal,
        peakAmplitude: shVal * distanceGain
      });
    }
    
    return channels;
  }
};

// Main SHAC visualization component
const SHACVisualization = () => {
  // State for visualization parameters
  const [ambisonicOrder, setAmbisonicOrder] = useState(3);
  const [azimuth, setAzimuth] = useState(0);
  const [elevation, setElevation] = useState(0);
  const [distance, setDistance] = useState(2);
  const [ambisonicChannels, setAmbisonicChannels] = useState([]);
  
  // Canvas reference for 3D visualization
  const canvasRef = useRef(null);
  
  // Generate artificial audio data
  const audioData = mockSHAC.generateAudioData(440, 2, 44100);
  
  // Update ambisonic channels when parameters change
  useEffect(() => {
    const position = [azimuth, elevation, distance];
    const channels = mockSHAC.encodeMonoSource(audioData, position, ambisonicOrder);
    setAmbisonicChannels(channels);
    
    // Update 3D visualization
    renderSoundField();
  }, [ambisonicOrder, azimuth, elevation, distance]);
  
  // Function to render 3D sound field visualization
  const renderSoundField = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(centerX, centerY) - 20;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw coordinate system
    ctx.strokeStyle = '#ccc';
    ctx.beginPath();
    ctx.moveTo(centerX - radius, centerY);
    ctx.lineTo(centerX + radius, centerY);
    ctx.moveTo(centerX, centerY - radius);
    ctx.lineTo(centerX, centerY + radius);
    ctx.stroke();
    
    // Draw listener (center point)
    ctx.fillStyle = 'blue';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw source position
    const sourceX = centerX + Math.sin(azimuth) * Math.cos(elevation) * radius / distance;
    const sourceY = centerY - Math.sin(elevation) * radius / distance;
    
    ctx.fillStyle = 'red';
    ctx.beginPath();
    ctx.arc(sourceX, sourceY, 8, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw line from listener to source
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(sourceX, sourceY);
    ctx.stroke();
    
    // Draw sound field energy
    const resolution = 20;
    const angleStep = (2 * Math.PI) / resolution;
    
    for (let a = 0; a < 2 * Math.PI; a += angleStep) {
      for (let e = -Math.PI/2; e <= Math.PI/2; e += angleStep) {
        const fieldStrength = ambisonicChannels.reduce((sum, channel) => {
          return sum + mockSHAC.sphericalHarmonic(channel.degree, channel.order, a, e) * channel.peakAmplitude;
        }, 0);
        
        const normalizedStrength = Math.max(0, Math.min(1, (fieldStrength + 1) / 2));
        const x = centerX + Math.sin(a) * Math.cos(e) * radius * normalizedStrength;
        const y = centerY - Math.sin(e) * radius * normalizedStrength;
        
        // Draw energy point
        ctx.fillStyle = `rgba(255, 165, 0, ${normalizedStrength})`;
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    
    // Draw orientation indicator
    ctx.strokeStyle = 'blue';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + Math.cos(0) * 20, centerY - Math.sin(0) * 20); // Forward direction
    ctx.stroke();
    
    // Add labels
    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.fillText('Front', centerX + radius + 5, centerY);
    ctx.fillText('Left', centerX, centerY - radius - 10);
    ctx.fillText('Right', centerX, centerY + radius + 15);
    ctx.fillText('Back', centerX - radius - 30, centerY);
  };
  
  // Convert channel data to chart format
  const channelChartData = ambisonicChannels.map(channel => ({
    name: `(${channel.degree},${channel.order})`,
    amplitude: Math.abs(channel.peakAmplitude),
    coefficient: channel.coefficient
  }));
  
  return (
    <div className="shac-visualization p-4 bg-gray-50 rounded-lg">
      <h1 className="text-2xl font-bold mb-4 text-center">Spherical Harmonic Audio Codec Visualization</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-xl font-semibold mb-3">3D Sound Field</h2>
          <canvas 
            ref={canvasRef} 
            width={400} 
            height={400} 
            className="bg-gray-100 rounded"
          />
          
          <div className="mt-4">
            <div className="mb-2">
              <label className="block text-sm font-medium text-gray-700">Ambisonic Order: {ambisonicOrder}</label>
              <input
                type="range"
                min={1}
                max={5}
                value={ambisonicOrder}
                onChange={(e) => setAmbisonicOrder(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-sm font-medium text-gray-700">Azimuth: {azimuth.toFixed(2)} rad</label>
                <input
                  type="range"
                  min={-Math.PI}
                  max={Math.PI}
                  step={0.1}
                  value={azimuth}
                  onChange={(e) => setAzimuth(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700">Elevation: {elevation.toFixed(2)} rad</label>
                <input
                  type="range"
                  min={-Math.PI/2}
                  max={Math.PI/2}
                  step={0.1}
                  value={elevation}
                  onChange={(e) => setElevation(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
            
            <div className="mt-2">
              <label className="block text-sm font-medium text-gray-700">Distance: {distance.toFixed(1)} meters</label>
              <input
                type="range"
                min={0.5}
                max={10}
                step={0.1}
                value={distance}
                onChange={(e) => setDistance(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-xl font-semibold mb-3">Ambisonic Channel Coefficients</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={channelChartData}
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 5
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="amplitude" 
                  stroke="#8884d8" 
                  activeDot={{ r: 8 }} 
                  name="Channel Amplitude"
                />
                <Line 
                  type="monotone" 
                  dataKey="coefficient" 
                  stroke="#82ca9d" 
                  name="SH Coefficient"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <h3 className="text-lg font-semibold mt-4 mb-2">Channel Information</h3>
          <div className="max-h-64 overflow-y-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Channel</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Degree (l)</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Order (m)</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Amplitude</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {ambisonicChannels.map((channel) => (
                  <tr key={channel.channel}>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900">{channel.channel}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900">{channel.degree}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900">{channel.order}</td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900">{channel.peakAmplitude.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      <div className="mt-6 bg-white p-4 rounded shadow">
        <h2 className="text-xl font-semibold mb-3">How SHAC Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <h3 className="text-md font-medium">1. Source Encoding</h3>
            <p className="text-sm text-gray-600">
              SHAC encodes each mono audio source into a set of ambisonic channels using spherical harmonics.
              The source position (azimuth, elevation, distance) determines the encoding coefficients.
            </p>
          </div>
          <div>
            <h3 className="text-md font-medium">2. Sound Field Processing</h3>
            <p className="text-sm text-gray-600">
              The ambisonic representation allows efficient rotation, translation, and manipulation of the entire sound field.
              Higher orders provide better spatial resolution.
            </p>
          </div>
          <div>
            <h3 className="text-md font-medium">3. Binaural Rendering</h3>
            <p className="text-sm text-gray-600">
              For headphone playback, SHAC applies HRTF convolution to convert the ambisonic signals into stereo audio,
              creating a realistic 3D audio experience.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SHACVisualization;