import * as siteConfig from './content/site/info.json'

export default {
  // Target (https://go.nuxtjs.dev/config-target)
  target: 'static',

  content: {
    prism: {
      theme: 'prism-themes/themes/prism-dark.css'
    },
    markdown: {
      remarkPlugins: [
        'remark-math'
      ],
      rehypePlugins: [
        'rehype-katex'
      ]
    }
  },

  // Environment variables: https://nuxtjs.org/api/configuration-env/
  env: {
    url:
      process.env.NODE_ENV === 'production'
        ? process.env.URL || 'http://createADotEnvFileAndSetURL'
        : 'http://localhost:3000',
    lang: 'en-US',
  },

  // Global page headers (https://go.nuxtjs.dev/config-head)
  head: {
    title: siteConfig.sitename || process.env.npm_package_name || '',
    meta: [
      { charset: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      {
        hid: 'description',
        name: 'description',
        content:
          siteConfig.sitedescription ||
          process.env.npm_package_description ||
          '',
      },
      { name: 'twitter:card', content: 'summary' },
      { name: 'twitter:site', content: '@shusukeioku' },
      { name: 'twitter:title', content: 'shusukeioku' },
      { name: 'twitter:description', content: "Shusuke Ioku's Web Page" },
    ],
    link: [
      { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
      { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.11.0/dist/katex.min.css' },
      { rel: "preconnect", href: "https://fonts.googleapis.com" },
      { rel: 'preconnect', href: "https://fonts.gstatic.com", crossorigin},
      { href: 'https://fonts.googleapis.com/css2?family=Noto+Serif+JP&family=Ubuntu:wght@400;700&display=swap', rel: 'stylesheet' }
    ],
  },

  generate: {
    fallback: true,
    exclude: [
      /^\/admin/, // path starts with /admin
    ],
  },

  // Global CSS (https://go.nuxtjs.dev/config-css)
  css: ['@/assets/css/main.css'],

  // Plugins to run before rendering page (https://go.nuxtjs.dev/config-plugins)
  plugins: [],

  // Auto import components (https://go.nuxtjs.dev/config-components)
  components: true,

  // Modules for dev and build (recommended) (https://go.nuxtjs.dev/config-modules)
  buildModules: [
    // https://go.nuxtjs.dev/eslint
    //'@nuxtjs/eslint-module',
    // https://go.nuxtjs.dev/tailwindcss
    '@nuxtjs/tailwindcss',
  ],

  // Modules (https://go.nuxtjs.dev/config-modules)
  modules: [
    // https://go.nuxtjs.dev/content
    '@nuxt/content',
  ],
}
