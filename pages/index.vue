<template>
  <div>
    <div class="flex flex-wrap w-fit justify-around">
      <div class="text-base intro" lang="en">
        <h2 class="text-3xl text-white font-bold text-center mt-12">Hi.</h2>
        <p>
          My name is 井奥崇輔. I study politics with a particular focus on interstate violence and state formation. I recieved an B.A. (Philosophy) from the University of Tokyo and an M.A. (Political Science) from Waseda University.
        </p>
        <p>
          My main project is the formal modeling and structural estimation of <span class="text-white font-bold">dynamic coercive diplomacy</span>. Previous theoretical studies of interstate conflict typically have dealt with coercive diplomacy as a one-shot game. I instead propose a dynamic model of coercive bargaining that captures the interaction as an infinite sequence of signaling/learning continuing on a "gray zone" between peace and military conflict. This yields novel insights into the strategic rationality of low-level violence between adversaries, alliance credibility, and the sources of dangerous interstate crises.
        </p>
        <p>
          Another set of my projects aims to unpack the process of <span class="text-white font-bold">state formation in Japan</span> (15th-19th Century), focusing on the causal effect of ingestion of Western advanced military technologies &mdash; how 鉄炮 (guns) changed battles during the Sengoku period, and how Western threats fostered the rapid modernization since the end of Edo period.
        </p>
        <p>
          I also work on a joint data collection project with Prof. Shuhei Kurizaki and Rui Asano about the emergence of and changes in diplomatic mission exchange practice between sovereign states.
        </p>
      </div>
      <div>
        <h2 class="text-3xl text-white font-bold text-center mt-12">Posts</h2>
        <div class="lg:grid lg:grid-cols-1 lg:gap-8">
          <div
            v-for="(post, index) in posts"
            :key="index"
            class="py-2 box"
          >
          <div class="flex">
            <div class="shrink-0 mr-2">
              <img :src="post.image" class="w-16 h-16 rounded-full border-gray border-2">
            </div>
            <div>
              <h3 class="text-white text-lg">
              <nuxt-link :to="`/blog/${post.slug}`">{{ post.title }}</nuxt-link>
              </h3>
              <p class="py-2">
                {{ post.date.substring(0,10) }}
              </p>
            </div>
          </div>
            <p>
              {{ post.description }}
            </p>
          </div>
        </div>
        <p class="text-center mt-8">
          <nuxt-link
            to="/blog"
            class="inline-flex items-center px-6 py-3 text-base font-medium leading-6 transition duration-150 ease-in-out bg-gray border border-gray-300 rounded-md text-white hover:text-orange-500 focus:outline-none focus:border-blue-300 focus:shadow-outline-blue active:text-gray-800 active:bg-gray-50"
          >
            More
          </nuxt-link>
        </p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  async asyncData({ $content }) {
    const page = await $content('index').fetch()
    const posts = await $content('blog').fetch()

    return {
      page,
      posts,
    }
  },
}
</script>

<style lang="css">
.intro {
  max-width: 500px;
  width: 99%;
}
.intro p {
  padding: .5em 0;
}
.box {
  max-width: 500px;
  width: 99%;
}
</style>
