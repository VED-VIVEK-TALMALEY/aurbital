import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "#040711",
        panel: "#0a1222",
        panel2: "#101c33",
        accent: "#00d4ff",
      },
    },
  },
  plugins: [],
} satisfies Config;
