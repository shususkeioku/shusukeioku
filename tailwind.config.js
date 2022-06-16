module.exports = {
  theme: {
    extend: {
      colors: {
        kaldi: '#ff4400',
        regalia: '#522d80'
      }
    },
  },
  plugins: [
    require("tailwindcss-hyphens"),
    require('@tailwindcss/typography'),
    // ...
  ],
}
