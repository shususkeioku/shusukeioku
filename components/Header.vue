<template>
  <header class="flex mx-auto mb-12 icon">
    <img src="/img/icon.jpg" class="w-20 h-20 rounded-full shrink-0">
    <div class="ml-5">
      <nuxt-link to="/">
        <h1 class="m-0 text-base text-white">
          Shusuke Ioku
        </h1>
        <p class="m-0 text-xs">
          Ph.D. student, Department of Political Science, University of Rochester
        </p>
      </nuxt-link>
      <p class="text-xs">
        CV | Projects | Teaching | Posts
      </p>
    </div>
  </header>
</template>

<script>
export default {}
</script>
<style scoped>
.icon {
  max-width: 400px;
  width: 95%;
}
</style>
