from django.core.management.base import BaseCommand
from django.db import transaction
from analytics.models import Prediction, HealthZone, ClinicalData
from analytics.disease_analyzer import DiseaseAnalyzer
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Backfill propagation_rate for existing predictions using historical data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be updated without making changes',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of predictions to process per batch (default: 100)',
        )
        parser.add_argument(
            '--zone',
            type=str,
            help='Optional: limit to a specific health zone name',
        )
        parser.add_argument(
            '--disease',
            type=str,
            help='Optional: limit to a specific disease name',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        batch_size = options['batch_size']
        zone_filter = options.get('zone')
        disease_filter = options.get('disease')

        self.stdout.write(self.style.SUCCESS('🚀 Starting propagation_rate backfill...'))

        # Build queryset for predictions needing update
        qs = Prediction.objects.all()
        if zone_filter:
            qs = qs.filter(health_zone__name__iexact=zone_filter)
        if disease_filter:
            qs = qs.filter(disease_name__iexact=disease_filter)

        # Get distinct (health_zone, disease_name) pairs
        pairs = qs.values('health_zone', 'disease_name').distinct()
        total_pairs = pairs.count()
        self.stdout.write(f"Found {total_pairs} unique (zone, disease) pairs to process")

        if dry_run:
            self.stdout.write(self.style.WARNING("🔍 DRY-RUN mode — no changes will be saved"))

        updated_count = 0
        error_count = 0
        skipped_count = 0

        for idx, pair in enumerate(pairs, 1):
            health_zone_id = pair['health_zone']
            disease = pair['disease_name']

            try:
                zone = HealthZone.objects.get(id=health_zone_id)

                # Calculate propagation
                propagation_rate = DiseaseAnalyzer.calculate_propagation_rate(
                    health_zone=zone,
                    disease_name=disease,
                    months_window=6
                )

                # Get predictions affected
                pred_qs = Prediction.objects.filter(
                    health_zone=zone,
                    disease_name=disease
                )
                if zone_filter or disease_filter:
                    # Re-apply filters if any
                    if zone_filter:
                        pred_qs = pred_qs.filter(health_zone__name__iexact=zone_filter)
                    if disease_filter:
                        pred_qs = pred_qs.filter(disease_name__iexact=disease_filter)

                count = pred_qs.count()

                if dry_run:
                    self.stdout.write(
                        f"  [{idx}/{total_pairs}] {zone.name} / {disease}: "
                        f"propagation={propagation_rate}% (would update {count} rows)"
                    )
                else:
                    rows_updated = pred_qs.update(propagation_rate=propagation_rate)
                    self.stdout.write(
                        f"  ✓ [{idx}/{total_pairs}] {zone.name} / {disease}: "
                        f"propagation={propagation_rate}% ({rows_updated} rows updated)"
                    )
                    updated_count += count

            except HealthZone.DoesNotExist:
                error_count += 1
                self.stdout.write(
                    self.style.ERROR(f"  ✗ [{idx}/{total_pairs}] Zone ID {health_zone_id} not found")
                )
            except Exception as e:
                error_count += 1
                self.stdout.write(
                    self.style.ERROR(f"  ✗ [{idx}/{total_pairs}] Error on {health_zone_id}/{disease}: {e}")
                )
                skipped_count += count if 'count' in locals() else 0

        # Summary
        self.stdout.write("")
        if dry_run:
            self.stdout.write(self.style.SUCCESS(
                f"📊 Dry-run complete. Would update ~{updated_count} predictions "
                f"across {total_pairs} zone/disease pairs."
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"✅ Backfill complete! Updated {updated_count} prediction rows "
                f"across {total_pairs} zone/disease pairs."
            ))

        self.stdout.write(f"   Errors: {error_count}")

        if skipped_count > 0 and not dry_run:
            self.stdout.write(f"   Rows skipped due to errors: {skipped_count}")
