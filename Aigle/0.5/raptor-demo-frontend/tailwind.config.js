/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        navy: {
          950: '#0d1628',
          900: '#111e35',
          800: '#162540',
          700: '#1b2d4e',
          600: '#21365c',
        },
      },
      boxShadow: {
        'glow-blue': '0 0 24px rgba(59,130,246,0.3)',
        'glow-blue-sm': '0 0 12px rgba(59,130,246,0.18)',
        'card': '0 1px 3px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.055)',
        'card-hover': '0 6px 16px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.09)',
      },
    },
  },
  plugins: [],
}
