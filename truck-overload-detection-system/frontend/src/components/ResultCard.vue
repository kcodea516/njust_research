<template>
  <article class="result-card">
    <span>{{ title }}</span>
    <strong>{{ value }}</strong>
    <small v-if="meta">{{ meta }}</small>
  </article>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true,
  },
  value: {
    type: String,
    required: true,
  },
  meta: {
    type: String,
    default: "",
  },
});
</script>
