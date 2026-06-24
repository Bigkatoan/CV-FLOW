/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        node: {
          input:   '#1e3a5f',
          process: '#1a3d2b',
          spatial: '#3d2b1a',
          utility: '#2d1a3d',
          output:  '#3d1a1a',
          cpp:     '#1a2d3d',
        },
      },
    },
  },
  plugins: [],
}
