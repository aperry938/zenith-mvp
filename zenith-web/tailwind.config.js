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
                    cyan: '#00ccff',
                }
            },
            fontFamily: {
                mono: ['Courier New', 'monospace'],
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                scanline: {
                    '0%': { transform: 'translateY(0)' },
                    '100%': { transform: 'translateY(100%)' },
                },
            },
            animation: {
                fadeIn: 'fadeIn 0.3s ease-out',
                scanline: 'scanline 4s linear infinite',
            }
        },
    },
    plugins: [],
}
