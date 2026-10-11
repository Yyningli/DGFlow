import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import * as format from '../src/lib/format.js'

test('protocol timing data excludes nested timings from the stage subtotal', () => {
    const split = format.splitStageTimes({
      dkg_s: 1, client_stage_s: 3, validation_s: 4, aggregation_s: 12,
      validation_key_s: 1, authorization_s: 3, aggregate_key_s: 2,
      partial_decryption_s: 3, aggregate_verification_s: 1,
      combine_and_confirmation_s: 5, combine_metrics_s: 1,
      combine_total_s: 4, combine_proof_verification_s: 2,
      proof_wall_sum_s: 8, combine_total_s_max: 9, combine_cpu_s: 10,
    })
    assert.equal(split.stages.reduce((total, row) => total + row.value, 0), 20)
    assert.equal(split.stages.length, 4)
    assert.equal(split.subStages.find(row => row.key === 'combine_and_confirmation_s').value, 5)
    assert.equal(split.subStages.filter(row => row.parent === 'aggregation_s').length, 5)
    assert.equal(split.clientSums[0].value, 8)
    assert.equal(split.combineParts.find(row => row.key === 'combine_total_s').value, 4)
    const orphan = format.splitStageTimes({ authorization_s: 3 })
    assert.equal(orphan.stages.length, 0)
    assert.equal(orphan.subStages.length, 1)
})

test('exported historical preparation and commit timings retain their aggregation parent', () => {
  const historical = JSON.parse(readFileSync(new URL('./fixtures/stage-times-prepare-commit.json', import.meta.url), 'utf8'))
    const split = format.splitStageTimes(historical.stage_times)
    assert.ok(Math.abs(split.stages.reduce((total, row) => total + row.value, 0) - historical.expected_stage_subtotal_s) < 1e-8)
    assert.equal(split.stages.find(row => row.key === 'aggregation_s').value, historical.stage_times.aggregation_s)
    for (const key of ['aggregate_prepare_s', 'aggregate_commit_s', 'aggregate_metrics_s']) {
      assert.equal(split.subStages.find(row => row.key === key && row.parent === 'aggregation_s').value, historical.stage_times[key])
    }
})
