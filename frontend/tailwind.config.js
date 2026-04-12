/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      colors: {
        base: '#0B0F14',
        card: '#11161D',
        'card-hover': '#161C26',
        'card-elevated': '#1A2130',
        border: '#1E2A3A',
        'border-hover': '#2A3A4E',
        primary: '#4DA3FF',
        'primary-hover': '#6BB5FF',
        'accent-green': '#22C55E',
        'accent-amber': '#F59E0B',
        'accent-rose': '#F43F5E',
        'text-primary': '#E6EDF3',
        'text-secondary': '#9BA7B4',
        'text-muted': '#5C6B7A',
      },
      borderRadius: {
        '2xl': '16px',
        'xl': '12px',
      },
      boxShadow: {
        'card': '0 2px 12px -4px rgba(0,0,0,0.3)',
        'card-hover': '0 4px 20px -4px rgba(0,0,0,0.4)',
        'hero': '0 4px 24px -4px rgba(0,0,0,0.5)',
        'glow-primary': '0 0 20px -4px rgba(77,163,255,0.35)',
      },
      transitionTimingFunction: {
        'premium': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
      animation: {
        'fade-in': 'fadeIn 0.35s cubic-bezier(0.25,0.1,0.25,1) both',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
