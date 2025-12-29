/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                zenith: {
                    bg: '#050505',
                    panel: '#0a0a0a',
                    neonBlue: '#00ccff',
                    neonPurple: '#d633ff',
                    neonGreen: '#00ff99',
                }
            },
            fontFamily: {
                mono: ['Courier New', 'monospace'],
            }
        },
    },
    plugins: [],
}
