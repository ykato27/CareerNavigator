/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // SkillNote-like colors (approximate)
        primary: "#00A968", // Green
        secondary: "#333333",
        background: "#F5F7F9",
      }
    },
  },
  plugins: [],
}
