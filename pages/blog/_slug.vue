<template>
  <div
    class="px-8 mx-auto mt-12 prose sm:px-6 md:px-4 lg:px-2 xl:px-0 xl:prose-xl lg:prose-lg md:prose-md"
  >
    <h2>{{ post.title }}</h2>
    <p>{{ post.date.substring(0,10) }}</p>
    <nuxt-content :document="post" />
    <h3>Tags</h3>
    <ul>
      <li v-for="(tag, index) in post.tags" :key="index">
        <nuxt-link :to="`/tags/${tag}`">
          {{ tag }}
        </nuxt-link>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  async asyncData({ $content, params: { slug } }) {
    const post = await $content('blog', slug).fetch()
    return {
      post,
    }
  },
  head() {
    return {
      meta: [
        { hid: 'og:description', property: 'og:description', content: this.post.description },
        { hid: 'og:url', property: 'og:url', content: this.$route.fullPath },
        { hid: 'og:image', property: 'og:image', content: this.post.image },
      ]
    }
  }
}
</script>

<style lang="css">
.prose {
  color: rgb(181, 180, 180)!important;
  max-width: 650px;
}

.prose a {
  color: rgb(181, 180, 180)!important;
}

.prose {
  font-size: 1rem/* 16px */;
  line-height: 1.75;
}

h2, h3 {
  color: white!important
}

a:visited {
  color: white!important
}
</style>
  
</style>
